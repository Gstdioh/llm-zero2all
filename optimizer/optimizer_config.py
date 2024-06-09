# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration for Optimizer."""

    ##############
    # Precision
    ##############
    precision_dtype: str = "bfloat16"
    """Train with precision_dtype mixed precision training. Defaults to False."""
    
    ###############
    # grad scaling
    ###############
    grad_scaling_before_comm: bool = True
    """
    是否在通信前进行梯度缩放，建议bfloat16下设为False，在最后除以值，减少精度损失
    
    False时，grad comm时不会scale，会在optim.step()中的前一刻scale，减少精度损失
    放在了复制梯度的操作中，_copy_model_grads_to_main_grads()
    """
    
    grad_scaling_factor: float = 1.0
    """
    相应的梯度缩放因子，grad_scaling_before_comm=False 下通常为 1.0 / tokens_per_iter
    
    grad_scaling_before_comm=True时，不起作用
    """

    ##############
    # overlap_optim_step
    ##############
    overlap_optim_step: bool = False
    """
    反向传播时，梯度通信时，会与优化器更新参数重叠
    因为通常通信时间比计算时间长，所以（梯度通信）与（backward计算和优化器更新）可以重叠
    
    optimizer需要构建self.bucket_to_optim_param_groups_map, self.bucket_to_model_params_map, self.bucket_to_main_params_map
    
    需要在optimizer中修改model中的comm_hook，添加一个overlap_optim_step_wrapper
    
    DistributedOptimize中需要构建：
        self.bucket_to_optim_param_groups_map,
        self.bucket_to_model_params_map, self.bucket_to_shard_model_params_map, self.bucket_to_shard_main_params_map
    
    注意，overlap_optim_step会改变grad_clip的语义
        因为原grad_clip会对所有参数的梯度计算一个norm，但是overlap_optim_step下是对每个bucket的梯度计算一个norm
        可能可以缓解参数原本的分布很不均匀，有的梯度大有的梯度小的问题？
    
    注意，float16下不能重叠，因为scaler中有cpu操作，会强制同步，同时会导致通信和backward计算也不重叠了，不推荐使用！
        不使用scaler的话，则可以
    
    同步：将step放在iter的最后执行
    
    重叠：将step放在comm_hook中执行，每次只执行一个bucket的step，计算和通信重叠
    """
    
    overlap_zero_grad_buffer: bool = False
    """
    是否在overlap_optim_step中将grad的清零也重叠起来
    
    若为False，则会对整个buffer进行清零（只启动一次kernel，可能更快）
    若为True，则会对每个bucket的buffer一个一个进行清零
    """
    
    grad_buffer_is_powerSGD_error: bool = False
    
    """
    梯度缓冲区是否是PowerSGD的error缓冲区，如果是，则不需要清零，这样可以节省内存
    """

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_param_gather: bool = False
    """
    If true, overlap param all-gather with forward compute in distributed optimizer.
    
    同步：见：parallel/distributed_optimizer/distributed_optimizer.py的step()下的
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
        同步情况下，会进行所有参数的all-gather操作
        
    重叠：见：parallel/distributed_optimizer/distributed_optimizer.py的enable_pre_hook()下的
        self.remove_pre_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(
            self._make_forward_pre_hook()
        )
        # 全局hook，对模型中的所有module都起作用
        # 假设有两个顺序module，module1, module2
        # 在module1进行forward前，会调用hook，对module2中的参数进行all-gather，计算与通信重叠
        # ! 注意，因为用的module.parameters(recurse=False)来遍历
        # !   所以model中最好不要使用nn.Parameter，否则可能会打断all-gather的流程
        
        第一次forward时，需要启动初始的all-gather，后续使用register_module_forward_pre_hook
        第一次启动all-gather见：parallel/distributed_optimizer/distributed_optimizer.py的zero_grad()下的
        # 这里会发起第一次的all-gather，后续的all-gather会在forward pre-hook中发起
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)
    """