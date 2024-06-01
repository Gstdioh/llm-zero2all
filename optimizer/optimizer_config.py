# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration for Optimizer."""

    ##############
    # normal
    ##############
    overlap_optim_step: bool = False
    """
    反向传播时，梯度通信时，会与优化器更新参数重叠
    因为通常通信时间比计算时间长，所以（梯度通信）与（backward计算和优化器更新）可以重叠
    
    optimizer需要构建self.bucket_to_optim_params_map
    
    同步：TODO_MY
    
    重叠：
    """

    ##############
    # Precision
    ##############
    precision_dtype: str = "bfloat16"
    """Train with precision_dtype mixed precision training. Defaults to False."""

    #######################
    # Distributed optimizer
    #######################
    use_distributed_optimizer: bool = False
    """Distribute optimizer state over data-parallel replicas."""

    overlap_grad_reduce: bool = False
    """
    If true, overlap grad reduce-scatter with backward compute in distributed optimizer.
    
    同步：设置bucket_size=None，即所有参数在同一个bucket中，计算完成后进行通信
    
    重叠：设置相应的bucket_size大小，分为多个bucket，每进行完一部分param的计算就进行异步通信，计算和通信重叠
    """

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