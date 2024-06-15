import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import random
import time

import torch
import torch.utils.data
import transformers
import utils
from utils import print_rank0

logger = logging.getLogger(__name__)

# 主进程才会输出信息
ddp = int(os.environ.get("RANK", -1)) != -1
master_process = True
ddp_rank = 0
if ddp:
    ddp_rank = int(os.environ["RANK"])
    master_process = ddp_rank == 0


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
    sources: Sequence[str],
    targets: Sequence[str],
    max_seq_len: int,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources_tokenized_list = [tokenizer.encode(text) for text in sources]
    targets_tokenized_list = [tokenizer.encode(text) for text in targets]
    
    # 记得首尾都添加个<|beginoftext|>的id，是tokenizer.begin_token_id
    add_token_id = tokenizer.my_begin_token_id
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


class SupervisedDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dir: str, max_seq_len: int, tokenizer: transformers.PreTrainedTokenizer, random_seed=42, **kwargs):
        super(SupervisedDataset, self).__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        
        print_rank0(logger.info, f"Loading data from {data_dir}...")
        file_path_list = utils.get_file_paths(data_dir, file_type="json")
        list_data_dict = []
        for file_path in file_path_list:
            data_list = json.load(open(file_path, "r", encoding="utf-8"))
            for data in data_list:
                list_data_dict.append(data)

        print_rank0(logger.info, "Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [example['output'] for example in list_data_dict]

        print_rank0(logger.info, "Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, self.max_seq_len, tokenizer)

        # 完整的长度（最大长度为max_seq_len + 1），后面获取的时候要进行截断，input_ids[i][:-1]，labels[i][1:]
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        
        # 样本数，样本的索引，通过索引来取数，对索引进行打乱
        self.num_samples = len(self.input_ids)
        self.sample_index_list = list(range(self.num_samples))
        
        # 提前进行一次打乱
        rng = random.Random(self.random_seed)
        rng.shuffle(self.sample_index_list)

    def __len__(self):
        return len(self.input_ids)

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
                
                x = self.input_ids[sample_index][:-1]
                y = self.labels[sample_index][1:]
                yield dict(
                    input_ids=x,
                    labels=y
                )
                
            # 下一个epoch，进行索引的打乱
            shuffle_time = time.time()
            rng.shuffle(sample_index_list)
            shuffle_time = time.time() - shuffle_time
            print_rank0(logger.info, f"worker_id {worker_id} ddp_rank {ddp_rank} shuffled {self.data_dir} sample_index_list, {shuffle_time:.4f}s")

            # 索引从头开始，即开始一个新的epoch
            cur_index %= self.num_samples


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.my_pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.my_pad_token_id),
        )
