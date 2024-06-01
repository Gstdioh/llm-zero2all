"""
Bucket定义见：parallel/distributed_data_parallel/param_and_grad_buffer.py

可能用到的参数：
self.param_data = param_data
self.grad_data = grad_data
"""

from typing import Any, Callable

import torch
import torch.distributed as dist


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int, data_parallel_rank: int):
    """
    返回指定的rank的shard（用于reduce_scatter_hook）
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    
    return buffer[(data_parallel_rank * shard_size) : ((data_parallel_rank + 1) * shard_size)]


def all_reduce_hook(bucket, process_group: dist.ProcessGroup=None, async_op=True):
    process_group = process_group if process_group is not None else dist.group.WORLD
    data_parallel_world_size = dist.get_world_size(process_group)
    
    gradient_scaling_factor = 1.0 / data_parallel_world_size
    bucket.grad_data *= gradient_scaling_factor
    
    communication_handle = dist.all_reduce(
        bucket.grad_data,
        group=process_group,
        async_op=async_op,
    )
    
    return communication_handle


def reduce_scatter_hook(bucket, process_group: dist.ProcessGroup=None, async_op=True):
    """
    用于分布式优化器，返回最后一个通信的handle，用于后续的同步
    """
    process_group = process_group if process_group is not None else dist.group.WORLD
    data_parallel_world_size = dist.get_world_size(process_group)
    data_parallel_rank = dist.get_rank(process_group)
    
    gradient_scaling_factor = 1.0 / data_parallel_world_size
    bucket.grad_data *= gradient_scaling_factor
    
    local_data_view = shard_buffer(bucket.grad_data, bucket.data_parallel_world_size, data_parallel_rank)
    communication_handle = dist._reduce_scatter_base(
        local_data_view,
        bucket.grad_data,
        group=process_group,
        async_op=async_op,
    )
    
    return communication_handle


def stream_wrapper(hook):
    """
    为hook添加一个stream，用于异步通信
    """
    def wrapper(*args, **kwargs):
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            event.wait(s)
            return hook(*args, **kwargs)
    return wrapper
