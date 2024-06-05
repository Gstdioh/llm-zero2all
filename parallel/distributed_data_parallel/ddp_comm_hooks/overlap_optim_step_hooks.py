"""
Bucket定义见：parallel/distributed_data_parallel/param_and_grad_buffer.py

可能用到的参数：
self.param_data = param_data
self.grad_data = grad_data

建议hook使用下面代码进行同步，然后hook最后用stream_wrapper包裹一下，以避免阻塞
event = torch.cuda.Event(enable_timing=False)
event.record(torch.cuda.current_stream())
return event
"""

import torch

from ..param_and_grad_buffer import Bucket


def _copy_bucket_optim_params_to_optim_groups(bucket: Bucket, optimizer):
    """
    将bucket中的参数对应的要更新的参数覆盖到optimizer的准备更新的groups中
    
    optimizer需要实现self.bucket_to_optim_param_groups_map
    即能将bucket映射到对应的shard的参数
    （FP32不需要，MixPrecision需要映射到完整参数，DistributedOptimizer需要映射到shard参数）
    
    self.bucket_to_optim_param_groups_map = {bucket: [group_params, ...]}
    同时构建bucket和shard optim param_groups的映射关系，用于overlap_optim_step
        bucket被唯一标识：id(bucket)
        shard optim param_groups： [group_params, ...], optimizer用于优化的参数组，
            元素是某个组的所有参数
    """
    for group_idx in range(len(optimizer.param_groups)):
        optimizer.param_groups[group_idx]["params"] = optimizer.bucket_to_optim_param_groups_map[bucket][group_idx]


def overlap_optim_step_wrapper(hook, optimizer):
    """
    为hook添加一个optimizer step，用于通信和参数更新的重叠
    
    返回event作为handle（有wait()方法）
    
    注意，需要CUDA_DEVICE_MAX_CONNECTIONS > 1，才能通信和参数更新重叠
    
    不需要使用stream考虑异步通信了，只需要最后再用stream_wrapper包裹一下即可
    """
    def wrapper(bucket: Bucket, *args, **kwargs):
        handle = hook(bucket, *args, **kwargs)
        
        handle.wait()
        
        _copy_bucket_optim_params_to_optim_groups(bucket, optimizer)
        
        optimizer.step(is_bucket_step=True, bucket=bucket)  # 指定bucket的step
        
        if optimizer.optim_config.overlap_zero_grad_buffer:
            bucket.grad_data.zero_()  # 清零bucket对应的model main_grads
            
        if bucket.is_last():
            # 记得将optim的param_groups还原回来
            for group_idx in range(len(optimizer.param_groups)):
                optimizer.param_groups[group_idx]["params"] = optimizer.all_optim_param_groups[group_idx]
        
        # 添加event，用于同步
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
            
        return event
        
    return wrapper
