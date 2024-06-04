"""
Bucket定义见：parallel/distributed_data_parallel/param_and_grad_buffer.py

可能用到的参数：
self.param_data = param_data
self.grad_data = grad_data
"""

from typing import Any, Callable

import torch
import torch.distributed as dist

from ..param_and_grad_buffer import Bucket


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int, data_parallel_rank: int):
    """
    返回指定的rank的shard（用于reduce_scatter_hook）
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    
    return buffer[(data_parallel_rank * shard_size) : ((data_parallel_rank + 1) * shard_size)]


def all_reduce_hook(bucket: Bucket, process_group: dist.ProcessGroup = None, async_op=True):
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


def reduce_scatter_hook(bucket: Bucket, process_group: dist.ProcessGroup = None, async_op=True):
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
    
    同时也能作为装饰器使用，能将hook异步执行，然后返回其handle（能使用wait()）
    
    建议每个hook最后都用这个包裹下，以避免阻塞
    """
    def stream_wrapper_hook(*args, **kwargs):
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            event.wait(s)
            hook(*args, **kwargs)
            event.record(torch.cuda.current_stream())
            return event
        
    return stream_wrapper_hook


def fp16_compress_wrapper(
    hook: Callable[[Any, Bucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, Bucket], torch.futures.Future[torch.Tensor]]:
    """
    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    floating point format (``torch.float16``), and casts the resulting tensor of the given hook back to
    the input data type, such as ``float32``.

    Therefore, ``fp16_compress_hook`` is equivalent to ``fp16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, fp16_compress_wrapper(powerSGD_hook))
        
    会阻塞，要异步请再用stream_wrapper包裹
    """

    def fp16_compress_wrapper_hook(
        bucket: Bucket, *args, **kwargs
    ) -> torch.futures.Future[torch.Tensor]:
        bucket.change_grad_buffer_dtype(torch.float16)
        
        handle = hook(bucket, *args, **kwargs)
        
        handle.wait()
        
        bucket.restore_grad_buffer_dtype()

        return handle

    return fp16_compress_wrapper_hook


def bf16_compress_wrapper(
    hook: Callable[[Any, Bucket], torch.futures.Future[torch.Tensor]]
) -> Callable[[Any, Bucket], torch.futures.Future[torch.Tensor]]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format> `_  (``torch.bfloat16``),
    and casts the resulting tensor of the given hook back to the input data type, such as ``float32``.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
        
    会阻塞，要异步请再用stream_wrapper包裹
    """

    def bf16_compress_wrapper_hook(
        bucket: Bucket, *args, **kwargs
    ) -> torch.futures.Future[torch.Tensor]:
        bucket.change_grad_buffer_dtype(torch.bfloat16)
        
        handle = hook(bucket, *args, **kwargs)
        
        handle.wait()
        
        bucket.restore_grad_buffer_dtype()

        return handle

    return bf16_compress_wrapper_hook
