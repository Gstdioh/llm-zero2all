import os
import copy
import logging
from typing import Dict, Optional, Sequence
import json
import random
import time
import tqdm
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.utils.data
import transformers
import utils
from utils import print_rank0, is_json_file
from utils.conversation import Conversation


logger = logging.getLogger(__name__)


IGNORE_INDEX = -100


def single_process_preprocess_conversation(
    args,
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
    max_prompt_len=None,
    prompt_truncation_mode="keep_end",
    user_name="human",
    assistant_name="gpt",
):
    process_id, all_data_list = args
    
    conv = Conversation(tokenizer=tokenizer)

    name_convert = {
        user_name: conv.roles_name[0],
        assistant_name: conv.roles_name[1]
    }
    
    data_dict = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
    }
    
    for i, row_data in tqdm.tqdm(enumerate(all_data_list), desc="Processing DPODataset", total=len(all_data_list), position=process_id):
        prompt = row_data["conversations"]
        chosen = row_data["chosen"]
        if not isinstance(chosen, list):
            chosen = [chosen]
        rejected = row_data["rejected"]
        if not isinstance(rejected, list):
            rejected = [rejected]
        
        # 保证第一个是human，去掉system，因为get_tokenized_conversations中会添加system
        while prompt[0]["from"] not in name_convert or name_convert[prompt[0]["from"]] != conv.roles_name[0]:
            prompt = prompt[1:]

        # tokenize prompt
        conv.clear()  # 清空messages
        for j, sentence in enumerate(prompt):
            role = name_convert[sentence["from"]]
            assert role == conv.roles_name[j % 2], f"conversations {i} get a wrong role {role} at turn {j}, expected {conv.roles_name[j % 2]} at turn {j}"
            conv.append_message(role, sentence["value"])
        # 构建prompt，并且进行tokenize，对于prompt不需要返回labels，后续全部进行mask
        prompt_output = conv.get_tokenized_conversations(return_labels=False, return_type="list")
        
        # tokenize chosen
        conv.clear()  # 清空messages
        for j, sentence in enumerate(chosen):
            role = name_convert[sentence["from"]]
            conv.append_message(role, sentence["value"])
        # 构建chosen，并且进行tokenize，会对labels进行掩码，不添加system和begin（prompt中添加过了）
        chosen_output = conv.get_tokenized_conversations(return_type="list", ignore_index=IGNORE_INDEX, add_begin=False, add_system=False)
        
        # tokenize rejected
        conv.clear()  # 清空messages
        for j, sentence in enumerate(rejected):
            role = name_convert[sentence["from"]]
            conv.append_message(role, sentence["value"])
        # 构建rejected，并且进行tokenize，会对labels进行掩码
        rejected_output = conv.get_tokenized_conversations(return_type="list", ignore_index=IGNORE_INDEX, add_begin=False, add_system=False)
        
        # 找出最长的回复，后面需要根据最长的回复进行截断
        longer_response_length = max(len(chosen_output["input_ids"]), len(rejected_output["input_ids"]))

        # 若整个长度（prompt + response）超过max_seq_len，先对prompt进行截断，prompt最长长度为max_prompt_len
        if len(prompt_output["input_ids"]) + longer_response_length > max_seq_len:
            if prompt_truncation_mode == "keep_start":
                prompt_output["input_ids"] = prompt_output["input_ids"][:max_prompt_len]
            elif prompt_truncation_mode == "keep_end":
                prompt_output["input_ids"] = prompt_output["input_ids"][-max_prompt_len:]
            else:
                raise ValueError(f"Invalid prompt_truncation_mode {prompt_truncation_mode}")
        
        # 若截断prompt后，还是超过max_seq_len，对response进行截断
        if len(prompt_output["input_ids"]) + longer_response_length > max_seq_len:
            for response in [chosen_output, rejected_output]:
                for key in response:
                    response[key] = response[key][:max_seq_len - len(prompt_output["input_ids"])]

        # 截断后，将prompt和response拼接，获取完整的input_ids和labels
        # labels的prompt部分全设为IGNORE_INDEX
        data_dict["chosen_input_ids"].append(torch.tensor(prompt_output["input_ids"] + chosen_output["input_ids"], dtype=torch.long))
        data_dict["chosen_labels"].append(torch.tensor(len(prompt_output["input_ids"]) * [IGNORE_INDEX] + chosen_output["labels"], dtype=torch.long))
        data_dict["rejected_input_ids"].append(torch.tensor(prompt_output["input_ids"] + rejected_output["input_ids"], dtype=torch.long))
        data_dict["rejected_labels"].append(torch.tensor(len(prompt_output["input_ids"]) * [IGNORE_INDEX] + rejected_output["labels"], dtype=torch.long))
    
    return data_dict


def preprocess_conversation(
    data_dir: str,
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
    save_cache_file_path=None,
    max_prompt_len=None,
    prompt_truncation_mode="keep_end",
    user_name="human",
    assistant_name="gpt",
    num_cpus=1,
    **kwargs,
) -> Dict:
    """
    多轮对话指令微调数据集的预处理，进行prompt构建、tokenize等操作
    
    max_prompt_len: prompt的最大长度，prompt+response的长度超过max_seq_len时，会根据这个来截断prompt
    prompt_truncation_mode: prompt的截断模式，keep_start或keep_end
    
    num_cpus: 并行处理的进程数，太大可能会卡住，设置为2还是会报错（不清楚为什么。。。）
        RuntimeError: unable to mmap 704 bytes from file <filename not specified>: Cannot allocate memory (12)
    
    all_data_list样例：
    [
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "Who are you?"
                },
                {
                    "from": "gpt",
                    "value": "I am a helpful assistant."
                },
                {
                    "from": "human",
                    "value": "Have a nice day!"
                },
                
            ],
            "chosen": {
                "from": "gpt",
                "value": "You too!"
            },
            "rejected": {
                "from": "gpt",
                "value": "Are you OK?"
            }
        },
        ...
    ]
    """
    # 最大长度需要加1，因为后面要进行截断，input_ids[i][:-1]，labels[i][1:]
    max_seq_len += 1
    
    save_cache_file_path = os.path.join(data_dir, f"cache_dpo_dataset_{max_seq_len}.pt")
    # 如果存在cache，则直接读取
    if os.path.exists(save_cache_file_path):
        print_rank0(logger.info, f"Loading data from cache {save_cache_file_path}...")
        return torch.load(save_cache_file_path)
    
    # 只有本地的主进程会进行数据的预处理
    if not utils.is_local_rank0():
        return None
    
    # 读取数据
    print_rank0(logger.info, f"Loading data from {data_dir}...")
    file_paths = utils.get_file_paths(data_dir, file_type="json")
    all_data_list = []
    for file_path in file_paths:
        if is_json_file(file_path):
            # 读json
            data_list = json.load(open(file_path, "r", encoding="utf-8"))
            all_data_list.extend(data_list)
        else:
            # 读jsonl
            with open(file_path, "r", encoding="utf-8") as f:
                # count = 100  # 测试用
                for line in f:
                    data = json.loads(line)
                    all_data_list.append(data)
                    # count -= 1
                    # if count == 0:
                    #     break
    
    if max_prompt_len is None:
        max_prompt_len = max_seq_len // 2
    
    single_process = partial(single_process_preprocess_conversation,
                             max_seq_len=max_seq_len,
                             tokenizer=tokenizer,
                             max_prompt_len=max_prompt_len,
                             prompt_truncation_mode=prompt_truncation_mode,
                             user_name=user_name,
                             assistant_name=assistant_name)
    
    if num_cpus > 1:
        with Pool(num_cpus) as p:
            all_data_list_split = utils.split_list_avg(all_data_list, num_cpus)
            data_dict_list = p.map(single_process, enumerate(all_data_list_split))
    else:
        data_dict_list = [single_process((0, all_data_list))]
    
    # 将所有的data_dict合并
    data_dict = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
    }
    for single_data_dict_list in data_dict_list:
        for key in data_dict:
            data_dict[key].extend(single_data_dict_list[key])
    
    # 保存cache
    print_rank0(logger.info, f"Save cache to {save_cache_file_path}")
    torch.save(data_dict, save_cache_file_path)

    return data_dict


class DPODataset(torch.utils.data.IterableDataset):
    """Dataset for Direct Preference Optimization."""

    def __init__(self,
                 data_dir: str,
                 max_seq_len: int,
                 tokenizer: transformers.PreTrainedTokenizer,
                 split="train",
                 valid_ratio=0.01,
                 random_seed=42,
                 **kwargs):
        """
        data_dir: 数据集所在的目录
        max_seq_len: 最大序列长度
        tokenizer: tokenizer
        random_seed: 打乱索引的随机种子
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.split = split
        self.valid_ratio = valid_ratio
        self.random_seed = random_seed

        # 多保留一个位置，因为后面要进行截断，input_ids[i][:-1]，labels[i][1:]
        save_cache_file_path = os.path.join(data_dir, f"cache_dpo_dataset_{max_seq_len + 1}.pt")

        print_rank0(logger.info, "Preprocessing DPODataset...")
        # 已经进行了截断，最大长度为max_seq_len + 1，后面获取的时候要进行截断，input_ids[i][:-1]，labels[i][1:]
        self.data_dict = preprocess_conversation(data_dir, self.max_seq_len, tokenizer, save_cache_file_path, **kwargs)
        
        # DDP下，其他进程要等待主进程预处理完数据，并保存了cache
        if utils.is_ddp():
            torch.distributed.barrier()
            
        # 其他进程从文件中读取
        if self.data_dict is None:
            self.data_dict = torch.load(save_cache_file_path)

        # 样本数，样本的索引，通过索引来取数，对索引进行打乱
        self.num_samples = len(self.data_dict["chosen_input_ids"])
            
        # 完整的样本索引列表
        self.sample_index_list = list(range(self.num_samples))
        
        # 进行数据集的划分
        self.train_num_samples = int(self.num_samples * (1 - self.valid_ratio))
        self.valid_num_samples = self.num_samples - self.train_num_samples

        # 提前进行一次打乱，打乱后再进行的划分
        rng = random.Random(self.random_seed)
        rng.shuffle(self.sample_index_list)

        print_rank0(logger.info, f"Loaded {self.data_dir} DPODataset, train_num_samples: {self.train_num_samples:,}, valid_num_samples: {self.valid_num_samples:,}")
        
        # 初始划分，用于判断当前使用的数据集
        self.split = "train"
        
        # 模式，["train", "eval"]，eval下不考虑DDP的影响
        self.mode = "eval"

    def train(self):
        """
        恢复原来的模式
        """
        self.mode = "train"
    
    def eval(self):
        """
        设置为eval模式，batch就不会按照ddp_world_size跳过了
        """
        self.mode = "eval"

    def get_single_sample(self, index):
        # 已经进行了截断，最大长度为max_seq_len + 1，这里进行截断，input_ids[i][:-1]，labels[i][1:]
        return dict(
            chosen_input_ids=self.data_dict["chosen_input_ids"][index][:-1],
            chosen_labels=self.data_dict["chosen_labels"][index][1:],
            rejected_input_ids=self.data_dict["rejected_input_ids"][index][:-1],
            rejected_labels=self.data_dict["rejected_labels"][index][1:],
        )

    def __iter__(self):
        # 若有多个worker（dataloader中的num_workers），则获取下信息
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        # 获取DDP信息
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 考虑数据集的划分
        if self.split == "train":
            sample_index_list = copy.deepcopy(self.sample_index_list[:self.train_num_samples])
        elif self.split == "valid":
            sample_index_list = copy.deepcopy(self.sample_index_list[self.train_num_samples:])
            
        # 验证模式下，不需要考虑ddp的影响，因为只有主进程会进行测试
        if self.mode == "eval":
            ddp_rank = 0
            ddp_world_size = 1
        
        # 后续打乱的随机种子
        rng = random.Random(self.random_seed + 1)
        
        cur_index = 0 + ddp_world_size * worker_id + ddp_rank  # 因为num_workers的数量可能大于batch_size，所以worker_id要在外层，ddp_rank要在内层
        all_index_len = num_workers * ddp_world_size  # cur_index每次应该跳过的长度
        
        # 无限循环，是否结束由外部控制
        while True:
            # 进行一个epoch的数据读取
            while cur_index < len(sample_index_list):
                sample_index = sample_index_list[cur_index]
                cur_index += all_index_len
                
                yield self.get_single_sample(sample_index)
                
            if self.split == "valid":
                # valid数据集只读取一次
                break
                
            # 下一个epoch，进行索引的打乱
            shuffle_time = time.time()
            rng.shuffle(sample_index_list)
            shuffle_time = time.time() - shuffle_time
            print_rank0(logger.info, f"worker_id {worker_id} ddp_rank {ddp_rank} shuffled {self.data_dir} sample_index_list, {shuffle_time:.4f}s")

            # 索引从头开始，即开始一个新的epoch
            cur_index %= len(sample_index_list)
            

class DataCollatorForDPODataset:
    """Collate examples for Direct Preference Optimization."""

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer is not None, "tokenizer must be provided"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        进行batch，padding，attention_mask
        """
        
        batch_samples = {
            "chosen_input_ids": [],
            "chosen_labels": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_labels": [],
            "rejected_attention_mask": [],
        }
        
        # batch，padding
        for key in ("chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"):
            # batch
            for instance in instances:
                batch_samples[key].append(instance[key])
                
            # padding
            if key.endswith("input_ids"):
                # input_ids用pad_token_id填充
                batch_samples[key] = torch.nn.utils.rnn.pad_sequence(
                    batch_samples[key], batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
            elif key.endswith("labels"):
                # labels用IGNORE_INDEX填充
                batch_samples[key] = torch.nn.utils.rnn.pad_sequence(
                    batch_samples[key], batch_first=True, padding_value=IGNORE_INDEX
                )

        # attention_mask
        for key in ("chosen_input_ids", "rejected_input_ids"):
            batch_samples[key.replace("input_ids", "attention_mask")] = batch_samples[key].ne(self.tokenizer.pad_token_id)
        
        return batch_samples
