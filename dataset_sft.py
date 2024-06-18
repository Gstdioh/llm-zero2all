import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import random
import time
import tqdm

import numpy as np
import torch
import torch.utils.data
import transformers
import utils
from utils import print_rank0
from sft import Conversation


logger = logging.getLogger(__name__)


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def preprocess(
    all_data_list,
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    单轮指令微调数据集的预处理，进行prompt构建、tokenize等操作
    
    all_data_list样例：
    [
        {
            "instruction": "根据下列信息，写一段描述。",
            "input": "地点: 青岛海滨\n情景: 晨间海浪拍岸，日出东升",
            "output": "青岛海滨的早晨，海浪轻拍着岸边，太阳东升，金光闪耀，景色宜人。"
        },
        ...
    ]
    """
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in all_data_list
    ]
    targets = [example['output'] for example in all_data_list]
    sources_tokenized_list = [tokenizer.encode(text) for text in sources]
    targets_tokenized_list = [tokenizer.encode(text) for text in targets]
    
    # 记得首尾都添加个<|beginoftext|>的id，是tokenizer.begin_token_id
    add_token_id = tokenizer.begin_token_id
    input_ids = [[add_token_id] + source_ids + target_ids + [add_token_id] for source_ids, target_ids in zip(sources_tokenized_list, targets_tokenized_list)]
    labels = copy.deepcopy(input_ids)
        
    # 将input_ids和labels转换为tensor，进行截断，只取前max_seq_len + 1个token，因为后面要进行截断：input_ids[i][:-1]，labels[i][1:]
    input_ids = [torch.tensor(input_id, dtype=torch.int64)[:max_seq_len + 1] for input_id in input_ids]
    labels = [torch.tensor(label, dtype=torch.int64)[:max_seq_len + 1] for label in labels]
    
    # 将instruction和input的部分mask掉，不计算损失
    for label, source_ids in zip(labels, sources_tokenized_list):
        source_len = len(source_ids) + 1  # 需要加一，因为前面添加了一个<|beginoftext|>
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


def preprocess_conversation(
    all_data_list,
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
    user_name="human",
    assistant_name="assistant",
) -> Dict:
    """
    多轮对话指令微调数据集的预处理，进行prompt构建、tokenize等操作
    
    all_data_list样例：
    [
        {
            "id": "identity_0",
            "conversations": [
                {
                    "from": "human",
                    "value": "Who are you?"
                },
                {
                    "from": "assistant",
                    "value": "I am a helpful assistant."
                },
                {
                    "from": "human",
                    "value": "Have a nice day!"
                },
                {
                    "from": "assistant",
                    "value": "You too!"
                }
            ]
        },
        ...
    ]
    """
    all_conversations = [data["conversations"] for data in all_data_list]
    
    conv = Conversation(tokenizer=tokenizer)

    name_convert = {
        user_name: conv.roles_name[0],
        assistant_name: conv.roles_name[1]
    }

    data_dict = {"input_ids": [], "labels": []}
    for i, conversations in tqdm.tqdm(enumerate(all_conversations), desc="Preprocessing sft dataset conversation", total=len(all_conversations)):
        if name_convert[conversations[0]["from"]] != conv.roles_name[0]:
            # Skip the first one if it is not from human
            conversations = conversations[1:]

        conv.clear()  # 清空messages
        for j, sentence in enumerate(conversations):
            role = name_convert[sentence["from"]]
            assert role == conv.roles_name[j % 2], f"conversations {i} get a wrong role {role} at turn {j}, expected {conv.roles_name[j % 2]} at turn {j}"
            conv.append_message(role, sentence["value"])
        
        # 构建prompt，并且进行tokenize，会对labels进行掩码
        output = conv.get_tokenized_prompt(ignore_index=IGNORE_INDEX, add_begin=True)
        
        
        data_dict["input_ids"].append(output["input_ids"][:max_seq_len + 1])
        data_dict["labels"].append(output["labels"][:max_seq_len + 1])
        
    return data_dict
    

class SupervisedDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 data_dir: str,
                 max_seq_len: int,
                 tokenizer: transformers.PreTrainedTokenizer,
                 sft_type="conversation",
                 use_dataset_with_index=True,
                 random_seed=42,
                 **kwargs):
        """
        data_dir: 数据集所在的目录
        max_seq_len: 最大序列长度
        tokenizer: tokenizer
        sft_type: 数据集类型，conversation: 多轮问答，其他: 单轮指令
        use_dataset_with_index：是否使用sample_index_map，节省内存，需要先对数据集进行预处理（pretokenize_sft_data.py），目前只支持conversation的预处理
        random_seed: 打乱索引的随机种子
        """
        super(SupervisedDataset, self).__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.sft_type = sft_type
        self.use_dataset_with_index = use_dataset_with_index
        self.random_seed = random_seed
        
        if use_dataset_with_index and sft_type == "conversation":
            print_rank0(logger.info, "Loading data from pretokenized_sft_data...")
            
            input_ids_file_paths = utils.get_file_paths(data_dir, file_type="bin", endswith="_input_ids.bin")
            self.input_ids_file_memmaps = []
            for file_path in input_ids_file_paths:
                # 文件可能为空，要考虑下
                try:
                    file_m = np.memmap(file_path, dtype=np.uint16, mode="r")
                except:
                    print_rank0(logger.warning, f"file {file_path} is empty, set np.array([]) instead.")
                    file_m = np.array([], dtype=np.uint16)
                self.input_ids_file_memmaps.append(file_m)
            
            labels_file_paths = utils.get_file_paths(data_dir, file_type="bin", endswith="_labels.bin")
            self.labels_file_memmaps = []
            for file_path in labels_file_paths:
                # 文件可能为空，要考虑下
                try:
                    file_m = np.memmap(file_path, dtype=np.uint16, mode="r")
                except:
                    print_rank0(logger.warning, f"file {file_path} is empty, set np.array([]) instead.")
                    file_m = np.array([], dtype=np.uint16)
                self.labels_file_memmaps.append(file_m)
                
            all_sample_index_map_path = utils.find_file_path(data_dir, "all_sample_index_map.ijson")[0]  # 找到的第一个文件
            self.sample_index_map = json.load(open(all_sample_index_map_path, "r", encoding="utf-8"))

            self.num_samples = len(self.sample_index_map)
        else:
            print_rank0(logger.info, f"Loading data from {data_dir}, need to preprocess data...")
            file_paths = utils.get_file_paths(data_dir, file_type="json")
            all_data_list = []
            for file_path in file_paths:
                try:
                    # 读json
                    data_list = json.load(open(file_path, "r", encoding="utf-8"))
                    all_data_list.extend(data_list)
                except:
                    # 读jsonl
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            all_data_list.append(data)

            print_rank0(logger.info, "Preprocessing SupervisedDataset...")
            if self.sft_type == "conversation":
                data_dict = preprocess_conversation(all_data_list, self.max_seq_len, tokenizer)
            elif self.sft_type == "instruction":
                data_dict = preprocess(all_data_list, self.max_seq_len, tokenizer)
            else:
                raise ValueError(f"Unsupported sft_type {self.sft_type}")

            # 完整的长度（最大长度为max_seq_len + 1），后面获取的时候要进行截断，input_ids[i][:-1]，labels[i][1:]
            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]
        
            # 样本数，样本的索引，通过索引来取数，对索引进行打乱
            self.num_samples = len(self.input_ids)
            
        # 样本索引列表
        self.sample_index_list = list(range(self.num_samples))
        
        # 提前进行一次打乱
        rng = random.Random(self.random_seed)
        rng.shuffle(self.sample_index_list)

    def __len__(self):
        return len(self.input_ids)

    def get_sample(self, index):
        if self.use_dataset_with_index and self.sft_type == "conversation":
            file_id, sample_start_offset, sample_len = self.sample_index_map[index]
            
            # 这里要进行截断，因为pretokenize时是对完整的文本进行tokenize的，没有进行max_seq_len的约束
            # 进行max_seq_len的截断，只取前max_seq_len + 1个token，要多取一个，因为后面要进行截断：input_ids[i][:-1]，labels[i][1:]
            sample_len = min(sample_len, self.max_seq_len + 1)
            
            input_ids = self.input_ids_file_memmaps[file_id][sample_start_offset:sample_start_offset + sample_len]
            input_ids = torch.from_numpy(input_ids.astype(np.int64))
            labels = self.input_ids_file_memmaps[file_id][sample_start_offset:sample_start_offset + sample_len]
            labels = torch.from_numpy(labels.astype(np.int64))

        else:
            # 这里不用进行截断，因为没有进行pretokenize，所以input_ids和labels是训练临时构建的（其中已经进行了截断，为了节省内存）
            input_ids = self.input_ids[index]
            labels = self.labels[index]
            
        return dict(
            input_ids=input_ids[:-1],
            labels=labels[1:],
        )

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        # get DDP rank info
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        rng = random.Random(self.random_seed + 1)
        sample_index_list = copy.deepcopy(self.sample_index_list)
        
        cur_index = 0 + ddp_world_size * worker_id + ddp_rank  # 因为num_workers的数量可能大于batch_size，所以worker_id要在外层，ddp_rank要在内层
        all_index_len = num_workers * ddp_world_size  # cur_index每次应该跳过的长度
        
        # 无限循环，是否结束由外部控制
        while True:
            # 进行一个epoch的数据读取
            while cur_index < self.num_samples:
                sample_index = sample_index_list[cur_index]
                cur_index += all_index_len
                
                yield self.get_sample(sample_index)
                
            # 下一个epoch，进行索引的打乱
            shuffle_time = time.time()
            rng.shuffle(sample_index_list)
            shuffle_time = time.time() - shuffle_time
            print_rank0(logger.info, f"worker_id {worker_id} ddp_rank {ddp_rank} shuffled {self.data_dir} sample_index_list, {shuffle_time:.4f}s")

            # 索引从头开始，即开始一个新的epoch
            cur_index %= self.num_samples
            

class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        assert self.tokenizer is not None, "tokenizer must be provided"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
