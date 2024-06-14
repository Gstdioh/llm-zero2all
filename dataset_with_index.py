import os
import random
import logging
import time

import numpy as np
import torch
import torch.utils.data

from utils import get_file_paths


logger = logging.getLogger(__name__)


# 主进程才会输出信息
ddp = int(os.environ.get("RANK", -1)) != -1
master_process = True
ddp_rank = 0
if ddp:
    ddp_rank = int(os.environ["RANK"])
    master_process = ddp_rank == 0


class PretokDataset(torch.utils.data.IterableDataset):
    """
    Loads pretokenized examples from disk and yields them as PyTorch tensors.
    
    self.split = "train" or "valid"
    
    先通过build_sample_index_map.py构建sample的索引，然后通过该索引来读取数据
    
    注意，要考虑到num_workers和ddp的影响
    """

    def __init__(self, split, max_seq_len, train_bin_dir, valid_bin_dir, random_seed=42):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.train_bin_dir = train_bin_dir
        self.valid_bin_dir = valid_bin_dir
        self.random_seed = random_seed
        
        self.bin_dir = self.train_bin_dir if self.split == "train" else self.valid_bin_dir
        
        self.file_list = get_file_paths(self.bin_dir, file_type="bin")
        self.file_memmaps = []
        for file in self.file_list:
            # 文件可能为空，要考虑下
            try:
                file_m = np.memmap(file, dtype=np.uint16, mode="r")
            except:
                _ = logger.warning(f"file {file} is empty, set np.array([]) instead.") if master_process else None
                file_m = np.array([], dtype=np.uint16)
            self.file_memmaps.append(file_m)
        
        self.sample_index_map_path = os.path.join(self.bin_dir, f"0_sample_index_map_{max_seq_len}.ibin")
        
        assert os.path.exists(self.sample_index_map_path), f"{self.sample_index_map_path} not exists, need to run build_sample_index_map.py first."
        
        # [(file_start_id, sample_start_offset), ...]
        # 保证每个样本的后面还有一个token可以取
        self.sample_index_map = np.memmap(self.sample_index_map_path, dtype=np.uint32, mode="r")
        self.num_samples = len(self.sample_index_map) // 2
        self.sample_index_list = list(range(self.num_samples))

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        # get DDP rank info
        ddp_rank = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        rng = random.Random(self.random_seed)
        
        cur_index = 0 + ddp_world_size * worker_id + ddp_rank  # 因为num_workers的数量可能大于batch_size，所以worker_id要在外层，ddp_rank要在内层
        all_index_len = num_workers * ddp_world_size  # cur_index每次应该跳过的长度
        
        while True:
            # 打乱索引
            shuffle_time = time.time()
            rng.shuffle(self.sample_index_list)
            shuffle_time = time.time() - shuffle_time
            print(f"worker_id {worker_id} ddp_rank {ddp_rank} shuffled {self.split} sample_index_list, {shuffle_time:.4f}s")
            
            cur_index %= self.num_samples  # 从头开始，即开始一个新的epoch
            
            while cur_index < self.num_samples:
                sample_index = self.sample_index_list[cur_index]
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
