import logging

import numpy as np
import torch
import torch.utils.data

from dataset import dataset_pretrain, dataset_sft, dataset_dpo


logger = logging.getLogger(__name__)


class Task:
    """
    生成训练时所需要的迭代器
    """
    
    def __init__(self, task_type, device, train_batch_size=None, valid_batch_size=None, num_workers=0, **dataset_kwargs):
        """
        dataset_kwargs: 传递给dataset的参数
            for pretrain:
                data_dir, max_seq_len, random_seed=42, use_dataset_with_index=False
            for sft:
                data_dir, max_seq_len, tokenizer, sft_type="conversation", random_seed=42
        """
        
        self.task_type = task_type
        self.device = device
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        
        # pin_memory: 如果是cpu则不需要pin_memory，其和non_blocking=True一块使用
        self.pin_memory = (device != "cpu")
        
        collect_fn = None
        # dataset
        if self.task_type == "pretrain":
            dataset = dataset_pretrain.get_pretrain_dataset(**dataset_kwargs)
        elif self.task_type == "sft":
            dataset = dataset_sft.SupervisedDataset(**dataset_kwargs)
            collect_fn = dataset_sft.DataCollatorForSupervisedDataset(**dataset_kwargs)
        elif self.task_type == "dpo":
            dataset = dataset_dpo.DPODataset(**dataset_kwargs)
            collect_fn = dataset_dpo.DataCollatorForDPODataset(**dataset_kwargs)
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
        self.dataset = dataset
        
        # dataloader，两个dataloader共用一个dataset
        self.train_dataloader = None
        self.valid_dataloader = None
        if self.train_batch_size is not None:
            self.train_dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=collect_fn, batch_size=train_batch_size,
                                                                num_workers=num_workers, pin_memory=self.pin_memory)
        if self.valid_batch_size is not None:
            self.valid_dataloader = torch.utils.data.DataLoader(self.dataset, collate_fn=collect_fn, batch_size=valid_batch_size,
                                                                num_workers=num_workers, pin_memory=self.pin_memory)
        
        self.train_num_samples = self.dataset.train_num_samples
        self.valid_num_samples = self.dataset.valid_num_samples

    def train(self):
        """
        设置为train模式，batch就会按照ddp_world_size跳过了
        """
        self.dataset.train()
    
    def eval(self):
        """
        设置为eval模式，batch就不会按照ddp_world_size跳过了
        """
        self.dataset.eval()
        
    def iter_batches(self, split, skip_batches=0):
        """
        根据dataloader生成迭代器，dataloader使用的dataset会无限生成样本
        
        split: 使用的数据集划分，确定当前生成器对应的划分
        skip_batches: 跳过的batch数量
        """
        # 设置数据集的划分，确保迭代数据集时用的是对应的索引列表
        # 如：train下：self.sample_index_list[:self.train_num_samples]
        self.dataset.split = split
        
        if split == "train":
            dataloader = self.train_dataloader
        else:
            dataloader = self.valid_dataloader
        
        # PyTorch会正确地在内部处理必要的同步，确保在计算操作（如线性层的计算）开始之前数据已经被传输完成。
        for batch in dataloader:
            # resume时，跳过已经训练过的batch
            if skip_batches > 0:
                skip_batches -= 1
                continue
            # 非阻塞的放入cuda中
            for key, value in batch.items():
                batch[key] = value.to(self.device, non_blocking=self.pin_memory)  # non_blocking=True和pin_memory=True一块使用
            yield batch
