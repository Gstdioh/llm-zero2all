import os
import random
from typing import List
import logging
import time
import copy

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist

from utils import get_file_paths, print_rank0


logger = logging.getLogger(__name__)


class PretokDataset(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    
    两个地方随机打乱，文件名和每个文件内部的一个样本
    """

    def __init__(self, data_dir, max_seq_len, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"rank {ddp_rank} created a PretokDataset with rng seed {seed}")
        
        shard_filenames = get_file_paths(self.data_dir, file_type="bin")
        assert len(shard_filenames) > 0, f"No bin files found in {self.data_dir}"
        
        while True:
            # 打乱文件
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                # 注意，data/02_train_data_more/01_bin_for_train_hf/file_004_00035.bin为空
                # 要考虑下文件可能为空的情况，直接跳过
                try:
                    m = np.memmap(shard, dtype=np.uint16, mode="r")
                except:
                    print(f"file {shard} is empty, skipping")
                    continue
                num_tokens = len(m)  # 这个shard的总token数
                num_batches = num_tokens // self.max_seq_len  # 向下取整
                # 太小不足以构成一个样本，跳过
                if num_batches <= 0:
                    print(f"file {shard} is too small, skipping")
                    continue
                # assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                # 打乱文件内的样本
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1  # 需要多一个token，因为y是x的下一个token
                    if end > num_tokens:
                        # 如果end越界，直接跳过，在num_tokens % self.max_seq_len == 0时会发生
                        continue
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield dict(
                        input_ids=x,
                        labels=y
                    )


class PretokDatasetWithIndex(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    
    先通过build_sample_index_map.py构建sample的索引，然后通过该索引来读取数据
    
    注意，要考虑到num_workers和ddp的影响
    """

    def __init__(self, data_dir, max_seq_len, random_seed=42, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.random_seed = random_seed
        
        self.file_path_list = get_file_paths(self.data_dir, file_type="bin")
        self.file_memmaps = []
        for file_path in self.file_path_list:
            # 文件可能为空，要考虑下
            try:
                file_m = np.memmap(file_path, dtype=np.uint16, mode="r")
            except:
                print_rank0(logger.warning, f"file {file_path} is empty, set np.array([]) instead.")
                file_m = np.array([], dtype=np.uint16)
            self.file_memmaps.append(file_m)
        
        self.sample_index_map_path = os.path.join(self.data_dir, f"0_sample_index_map_{max_seq_len}.ibin")
        
        assert os.path.exists(self.sample_index_map_path), f"{self.sample_index_map_path} not exists, need to run build_sample_index_map.py first."
        
        # [(file_start_id, sample_start_offset), ...]
        # 保证每个样本的后面还有一个token可以取
        self.sample_index_map = np.memmap(self.sample_index_map_path, dtype=np.uint32, mode="r")
        
        # 样本数，样本的索引，通过索引来取数，对索引进行打乱
        self.num_samples = len(self.sample_index_map) // 2
        self.sample_index_list = list(range(self.num_samples))
        
        # 提前进行一次打乱
        rng = random.Random(self.random_seed)
        rng.shuffle(self.sample_index_list)
        
    def __len__(self):
        return self.num_samples

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
                
                # 从sample_index_map中找到对应的sample_index
                file_start_id, sample_start_offset = tuple(self.sample_index_map[sample_index * 2: sample_index * 2 + 2])
                
                # 找从file_start_id, sample_start_offset开始的max_seq_len + 1的token，因为y是x的下一个token，需要多一个token
                chunk_list = []
                remain_tokens = self.max_seq_len + 1
                while len(self.file_memmaps[file_start_id]) - sample_start_offset < remain_tokens:
                    chunk_list.append(self.file_memmaps[file_start_id][sample_start_offset:])
                    remain_tokens -= len(self.file_memmaps[file_start_id]) - sample_start_offset
                    file_start_id += 1
                    sample_start_offset = 0
                chunk_list.append(self.file_memmaps[file_start_id][sample_start_offset: sample_start_offset + remain_tokens])
                chunk = np.concatenate(chunk_list, axis=0)
                chunk = torch.from_numpy(chunk.astype(np.int64))
                
                x = chunk[:-1]
                y = chunk[1:]
                yield dict(
                    input_ids=x,
                    labels=y
                )
                
            # 下一个epoch，进行索引的打乱
            shuffle_time = time.time()
            rng.shuffle(sample_index_list)
            shuffle_time = time.time() - shuffle_time
            print(f"worker_id {worker_id} ddp_rank {ddp_rank} shuffled {self.data_dir} sample_index_list, {shuffle_time:.4f}s")

            # 索引从头开始，即开始一个新的epoch
            cur_index %= self.num_samples
