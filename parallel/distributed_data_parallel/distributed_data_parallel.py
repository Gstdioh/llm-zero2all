# Adapted from Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from logging import getLogger
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn as nn

from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import ParamAndGradBuffer
from distributed import get_global_rank
from parallel.distributed_data_parallel.ddp_comm_hooks.default_hooks import all_reduce_hook, reduce_scatter_hook, stream_wrapper


logger = getLogger(__name__)


class DistributedDataParallel(nn.Module):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).
    
    模型的param和main_grad使用连续buffer来管理，参数使用id(param)作为key使用

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        data_parallel_group: Data-parallel process group.
        expert_data_parallel_group: Optional data-parallel process group for experts in a MoE.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        check_for_nan_in_grad: If true, check if local grad norm is NaN.

    """

    def __init__(
        self,
        module: torch.nn.Module,
        ddp_config: DistributedDataParallelConfig,
        data_parallel_group: torch.distributed.ProcessGroup = None,
        disable_bucketing: bool = False,
    ):
        super().__init__()
        self.module = module
        
        self.data_parallel_group = data_parallel_group if data_parallel_group is not None else torch.distributed.group.WORLD

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            dp_size = torch.distributed.get_world_size(self.data_parallel_group)
            ddp_config.bucket_size = max(40_000_000, 1_000_000 * dp_size)
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(f'Setting up DistributedDataParallel with config {self.ddp_config}')

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if disable_bucketing:
            self.bucket_size = None

        self.module = module
        self.param_to_buffer = {}

        # Group parameters by their gradient type.
        self.param_to_name = {}
        dense_params = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            param.grad_added_to_main_grad = False
            self.param_to_name[param] = name

            dense_params.append(param)

        def allocate_buffers_for_parameters(
            input_params, data_parallel_group,
        ):
            param_and_grad_dtype_to_params = {}

            # Group parameters by their gradient type.
            for param in input_params:
                if not param.requires_grad:
                    continue

                param_dtype = param.dtype
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

            # Allocate the grad buffers and map the grads.
            buffers = []
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    ParamAndGradBuffer(
                        self.ddp_config,
                        len(buffers),
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        self.param_to_name,
                    )
                )
                for param in params:
                    self.param_to_buffer[param] = buffers[-1]

            return buffers

        # Allocate the param+grad buffers for dense params' grads.
        # self.buffers: {(param_dtype, grad_dtype): params, ...}
        self.buffers = allocate_buffers_for_parameters(
            dense_params, self.data_parallel_group,
        )
        
        cur_bucket_id = 0  # int, bucket的全局id，在DDP设置完buffer后，会设置该值，倒序
        self.global_n_buckets = None  # int, DDP下所有bucket的数量，在DDP设置完buffer后，会设置该值
        # 设置每个bucket的self.global_bucket_id值
        # buffer是正序，bucket是倒序，所以需要这样计算，global bucket 按照倒序来
        for buffer in self.buffers[::-1]:
            for bucket in buffer.buckets:
                bucket.global_bucket_id = cur_bucket_id
                cur_bucket_id += 1
        self.global_n_buckets = cur_bucket_id
        # 设置所有bucket的self.global_n_buckets
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                bucket.global_n_buckets = self.global_n_buckets
        
        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        # 分布式优化器会重映射模型参数的param.data到连续buffer中，同时删除原param.data
        # 若有weight_tensor属性，会导致原param.data的引用还没有删除，导致显存并没有释放
        # 使用LayerNormLinear等时可能会存储一个weight_tensor属性
        if self.ddp_config.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
                self.grad_accs.append(grad_acc)
                
        # 同步模型参数
        self.broadcast_params()
            
        # 保存通信hook，和相应的状态，用于后续可能的overlap_optim_step_wrapper
        self.cur_comm_hook = None
        self.cur_comm_hook_args = []
        self.cur_comm_hook_kwargs = {}
        
        # 初始化通信hook，其中会保存通信hook，和相应的状态
        # Use async_op only when overlap_grad_reduce is True.
        if self.ddp_config.use_distributed_optimizer:
            self.register_comm_hook(stream_wrapper(reduce_scatter_hook), self.data_parallel_group, async_op=self.ddp_config.overlap_grad_reduce)
        else:
            self.register_comm_hook(stream_wrapper(all_reduce_hook), self.data_parallel_group, async_op=self.ddp_config.overlap_grad_reduce)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self,
        param: torch.nn.Parameter,
        param_to_buffer: Dict[torch.nn.Parameter, ParamAndGradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):
            if param.requires_grad:
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                # 见：megatron\core\tensor_parallel\layers.py的linear_with_grad_accumulation_and_async_allreduce
                # 其中实现了梯度累加的融合算子
                # 若使用了这个算子，则可以不用通过hook来累加梯度了（内部已经累加了）
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                # bucket就绪，则开始同步DP间的梯度
                # 考虑gradient accumulation，只有在最后一个microbatch时才会进行梯度同步
                # 使用is_last_microbatch来控制，见：megatron\core\distributed\param_and_grad_buffer.py的ParamAndGradBuffer
                # 注意，使用overlap_grad_reduce则会一个一个bucket进行同步；使用finish_grad_sync来去报异步通信完成
                if self.ddp_config.overlap_grad_reduce:
                    param_to_buffer[param].register_grad_ready(param)

        return param_hook
    
    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for buffer in self.buffers:
            buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for buffer in self.buffers:
                buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers:
            buffer.start_grad_sync()

    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers:
            buffer.scale_gradients(scaling_factor)

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for buffer in self.buffers:
            buffer.finish_grad_sync()

    def zero_grad_buffer(self, zero_grad_data: bool = True):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        
        overlap_optim_step会用到zero_grad_data=False，因为overlap_optim_step会自动清空梯度
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers:
            buffer.reset(zero_grad_data)

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        DDP进行init时会同步模型模型参数
        """
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data,
                src=get_global_rank(self.data_parallel_group, 0),  # 一个dp组中的local rank0对应的global rank，作为广播源
                group=self.data_parallel_group,
            )

    def state_dict(self, prefix='', keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)
    
    def register_comm_hook(self, hook: callable, *args, **kwargs):
        """
        Registers a communication hook for all the buffers.
        """
        for buffer in self.buffers:
            buffer.register_comm_hook(hook, *args, **kwargs)

        # 记录当前hook的信息，用于后续可能的overlap_optim_step_wrapper
        self.cur_comm_hook = hook
        self.cur_comm_hook_args = args if args is not None else []
        self.cur_comm_hook_kwargs = kwargs if kwargs is not None else {}
