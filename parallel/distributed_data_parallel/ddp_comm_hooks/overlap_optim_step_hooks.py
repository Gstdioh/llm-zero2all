import torch


def _copy_bucket_optim_params_to_optim_groups(bucket, optimizer):
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
    
    通过stream和event来同步通信和参数更新，最后返回event作为handle（有wait()方法）
    
    注意，需要CUDA_DEVICE_MAX_CONNECTIONS > 1，才能通信和参数更新重叠
    """
    def wrapper(bucket, *args, **kwargs):
        communication_handle = hook(bucket, *args, **kwargs)
        
        event = torch.cuda.Event(enable_timing=False)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            communication_handle.wait()
            
            _copy_bucket_optim_params_to_optim_groups(bucket, optimizer)
            
            optimizer.step(is_bucket_step=True, bucket=bucket)  # 指定bucket的step
            
            if optimizer.optim_config.overlap_zero_grad_buffer:
                bucket.grad_data.zero_()  # 清零bucket对应的model main_grads
            
            event.record(torch.cuda.current_stream())
            
        return event
        
    return wrapper


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