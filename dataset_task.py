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
    
    def __init__(self, task_type, batch_size, device, num_workers=0, **dataset_kwargs):
        """
        dataset_kwargs: 传递给dataset的参数
            for pretrain:
                data_dir, max_seq_len, random_seed=42, use_dataset_with_index=False
            for sft:
                data_dir, max_seq_len, tokenizer, sft_type="conversation", random_seed=42
        """
        
        self.task_type = task_type
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        
        # pin_memory: 如果是cpu则不需要pin_memory，其和non_blocking=True一块使用
        self.pin_memory = (device != "cpu")
        
        if self.task_type == "pretrain":
            dataset = dataset_pretrain.get_pretrain_dataset(**dataset_kwargs)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=self.pin_memory)
        elif self.task_type == "sft":
            dataset = dataset_sft.SupervisedDataset(**dataset_kwargs)
            collect_fn = dataset_sft.DataCollatorForSupervisedDataset(**dataset_kwargs)
            dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collect_fn, batch_size=batch_size, num_workers=num_workers, pin_memory=self.pin_memory)
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
                cur_batch[key] = value.to(self.device, non_blocking=self.pin_memory)  # non_blocking=True和pin_memory=True一块使用
            yield cur_batch
