import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist

from utils import get_file_paths
import dataset_pretrain
import dataset_sft


logger = logging.getLogger(__name__)


class Task:
    """
    生成训练时所需要的迭代器
    """
    
    def __init__(self, task_type, data_dir, max_seq_len, batch_size, device, num_workers=0, use_dataset_with_index=False, tokenizer=None):
        self.task_type = task_type
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.use_dataset_with_index = use_dataset_with_index
        
        if self.task_type == "pretrain":
            if use_dataset_with_index:
                dataset = dataset_pretrain.PretokDatasetWithIndex(data_dir, max_seq_len)
            else:
                dataset = dataset_pretrain.PretokDataset(data_dir, max_seq_len)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        elif self.task_type == "sft":
            assert tokenizer is not None, "tokenizer must be provided for supervised fine-tuning"
            dataset = dataset_sft.SupervisedDataset(data_dir, max_seq_len, tokenizer)
            collect_fn = dataset_sft.DataCollatorForSupervisedDataset(tokenizer)
            dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collect_fn, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
        self.dataset = dataset
        self.dataloader = dataloader
    
    def iter_batches(self, skip_batches=0):
        """
        根据dataloader生成迭代器，dataloader使用的dataset会无限生成样本
        """
        # PyTorch会正确地在内部处理必要的同步，确保在计算操作（如线性层的计算）开始之前数据已经被传输完成。
        for cur_batch in self.dataloader:
            # resume时，跳过已经训练过的batch
            if skip_batches > 0:
                skip_batches -= 1
                continue
            # 非阻塞的放入cuda中
            for key, value in cur_batch.items():
                cur_batch[key] = value.to(self.device, non_blocking=True)
            yield cur_batch
