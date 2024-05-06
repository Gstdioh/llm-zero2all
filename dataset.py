import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist

from utils import get_file_paths

DATA_CACHE_DIR = "data"


class PretokDataset(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    
    两个地方随机打乱，文件名和每个文件内部的一个样本
    """

    def __init__(self, split, max_seq_len, train_bin_dir, valid_bin_dir,):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.train_bin_dir = train_bin_dir
        self.valid_bin_dir = valid_bin_dir

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        
        # 训练集和验证集
        data_dir = self.train_bin_dir if self.split == "train" else self.valid_bin_dir
        shard_filenames = get_file_paths(data_dir, file_type="bin")
        assert len(shard_filenames)>0, f"No bin files found in {data_dir}"
        
        while True:
            # 打乱文件
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                # 打乱文件内的样本
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        # 使用pin_memory=True和non_blocking=True可以加速数据传输
        # 见：https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        # PyTorch会正确地在内部处理必要的同步，确保在计算操作（如线性层的计算）开始之前数据已经被传输完成。
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y
