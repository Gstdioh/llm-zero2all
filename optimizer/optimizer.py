import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod

import torch

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    multi_tensor_applier = None

from optimizer import OptimizerConfig
from parallel.distributed_data_parallel import DistributedDataParallel
from parallel.distributed_data_parallel.ddp_comm_hooks.default_hooks import stream_wrapper
from parallel.distributed_data_parallel.ddp_comm_hooks.overlap_optim_step_hooks import overlap_optim_step_wrapper
from parallel import Bucket


def zero_grad_group_helper(group: List[torch.nn.Parameter], set_to_none: bool):
    """
    Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer.
    """
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor], that: List[torch.Tensor], overflow_buf: Optional[torch.Tensor] = None
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf is not None and multi_tensor_applier is not None:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class Z2allOptimizer:
    """
    简单的包裹，使得访问Z2allOptimizer就像访问self.optimizer一样
    
    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        optim_config (OptimizerConfig): configuration object for optimizer.
        scaler: used for scaling gradients.
        grad_clip: used for cliping gradients.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, optim_config: OptimizerConfig,
                 scaler: torch.cuda.amp.GradScaler = None, grad_clip=0.0):
        self.optimizer = optimizer
        self.optim_config = optim_config
        self.scaler = scaler
        self.grad_clip = grad_clip
        
        self.update_successful = True  # optim参数更新成功后，才会进行all-gather，scaler时才会用到
        
        if not self.optim_config.use_distributed_optimizer:
            assert not self.optim_config.overlap_param_gather, "overlap_param_gather is only supported with distributed optimizer"
        
    def reset(self):
        """
        重置状态
        """
        self.update_successful = True
        
    def _post_init(self):
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self.all_params_for_unscaling_grads = [
            param for group in self.optimizer.param_groups for param in group["params"] 
            if param.requires_grad
        ]
        
    def __getattr__(self, name):
        """
        当访问NavieOptimizer的属性时，如果没有找到，就去optimizer中找
        
        相当于将NavieOptimizer视为optimizer的一个包装，对外部代码无影响
        """
        return getattr(self.optimizer, name)

    def _collect_main_params_for_unscaling_grads(self):
        """
        根据当前optim_param_groups收集需要unscale梯度的参数，即需要更新的参数，后续将其中的grads进行unscale

        overlap_optim_step下，收集的是bucket对应的参数，因为optim_param_groups会被更新为对应bucket的参数
        
        Note: this should be equivalent to the float-16 optimizer's method,
        but written differently, so the two should be combined.
        """
        return [
            param for group in self.optimizer.param_groups for param in group["params"]
        ]


class MixedPrecisionOptimizer(Z2allOptimizer, ABC):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        optim_config (OptimizerConfig): configuration object for optimizer.
        scaler: used for scaling gradients.
        grad_clip: used for cliping gradients.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, optim_config: OptimizerConfig,
                 scaler: torch.cuda.amp.GradScaler = None, grad_clip=0.0):
        Z2allOptimizer.__init__(self, optimizer, optim_config, scaler, grad_clip)

        # Dummy tensor needed for apex multi-apply tensor.
        # For bfloat, we don't have multi-tensor apply and for now
        # we set it to none so the multi-tensor apply gets ignored.
        # 用于加速多tensor操作的，对于bf16，暂时不支持，所以设置为None
        if self.optim_config.precision_dtype == "float16":
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        else:
            self._dummy_overflow_buf = None
            
    @abstractmethod
    def _copy_model_grads_to_main_grads(self):
        """
        将模型的梯度拷贝到优化器的梯度中，优化器中的梯度保存的是fp32的梯度，用于更新参数
        """
        pass
    
    @abstractmethod
    def _copy_main_params_to_model_params(self):
        """
        将优化器的参数拷贝到模型的参数中，优化器中的参数保存的是fp32的参数
        """
        pass
    
    @torch.no_grad()
    def step(self, is_bucket_step=False, bucket: Bucket = None):
        """
        更新参数，若需要则可以进行scale和grad_clip
        
        is_bucket_step, bucket用于overlap_optim_step
        
        1. 若overlap_optim_step=False，等待梯度同步完成，然后才能进行更新
        
        2. 更新参数，包括复制梯度到优化器的梯度中，更新参数，将optim参数覆盖到模型参数。
            非overlap_optim_step，直接更新所有参数
            overlap_optim_step，只有当is_bucket_step=True时，才会触发更新（在overlap_optim_step_wrapper包裹的comm_hook中进行更新）
        
        3. 若overlap_optim_step=True and not is_bucket_step，则等待参数更新完成
            是全部参数，进行等待，即在comm_hook中不会等，只会所有触发后才会等
            
        Return:
        update_successful (bool): True if the update was successful.
            若梯度中包含Nan，则不进行更新，逻辑在scaler.step中
        """
        if not self.optim_config.overlap_optim_step:
            # 1. 若overlap_optim_step=False，等待梯度同步完成
            for model_chunk in self.model_chunks:
                model_chunk.finish_grad_sync()
                
        if (not self.optim_config.overlap_optim_step) or (self.optim_config.overlap_optim_step and is_bucket_step):
            # Copy gradients from model params to main params.
            self._copy_model_grads_to_main_grads(is_bucket_step, bucket)
            
            if self.scaler is not None and self.scaler.is_enabled():
                # unscale，并且裁剪梯度
                if self.grad_clip != 0.0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self._collect_main_params_for_unscaling_grads(), self.grad_clip)
                # step the optimizer and scaler if training in fp16
                self.scaler.step(self.optimizer)  # 若没有unscale，进行unscale并且更新
                
                # 判断是否更新成功
                optimizer_state = self.scaler._per_optimizer_states[id(self.optimizer)]
                if sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                    # 有Nan或者inf
                    self.update_successful = False
                
                # overlap_optim_step中需要判断是否更改stage
                if self.optim_config.overlap_optim_step:
                    # 如果不是最后一个bucket，则需要更改stage为READY，即0，因为下次bucket又会用到scaler，防止报错
                    if not bucket.is_last():
                        optimizer_state["stage"] = 0  # 0表示READY
            else:
                # 裁剪梯度
                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self._collect_main_params_for_unscaling_grads(), self.grad_clip)
                self.optimizer.step()

            # Update params from optim params.
            # 更新失败的话，就不用覆盖模型参数了
            if self.update_successful:
                self._copy_main_params_to_model_params(is_bucket_step, bucket)
        
        if not is_bucket_step:
            # 每个iter只会调用一次，在这里更新scaler
            if self.scaler is not None and self.scaler.is_enabled():
                # 更新scaler的缩放因子，需要放在最后面，每个bucket更新时不需要对scaler进行更新，只需要最后更新一次即可
                # 为了进行下一个iter，会清空scale中optim的状态
                self.scaler.update()
        
            if self.optim_config.overlap_optim_step:
                # 等待参数更新完成
                for model_chunk in self.model_chunks:
                    model_chunk.finish_grad_sync()
            

class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        optim_config (OptimizerConfig): configuration object for optimizer.
        model_chunks: ddp model list.
        scaler: used for scaling gradients.
        grad_clip: used for cliping gradients.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 optim_config: OptimizerConfig,
                 model_chunks: Optional[Union[DistributedDataParallel, List[DistributedDataParallel]]],
                 scaler: torch.cuda.amp.GradScaler = None, grad_clip = 0.0):
        
        super().__init__(optimizer, optim_config, scaler, grad_clip)
        
        assert model_chunks is not None, "model is required!"
        if not isinstance(model_chunks, List):
            model_chunks = [model_chunks]
        self.model_chunks = model_chunks

        # overlap_optim_step需要实现self.bucket_to_optim_param_groups_map
        if self.optim_config.overlap_optim_step:
            # 这个重叠必须要有overlap_grad_reduce，要不然没意义了
            for model_chunk in model_chunks:
                assert model_chunk.ddp_config.overlap_grad_reduce, "ddp_config.overlap_grad_reduce is required for overlap_optim_step"

            # 1. 首先构建原model param到bucket的映射
            self.model_param_to_bucket_map = {}
            for model_chunk in model_chunks:
                for buffer in model_chunk.buffers:
                    for bucket in buffer.buckets:
                        for param in bucket.params:
                            self.model_param_to_bucket_map[param] = bucket
                            
            # 2. 构建bucket_to_optim_param_groups_map，遍历optim组
            # 代码逻辑与FP32的不太一样，因为optimizer的param_groups会重新构建，即将FP16 -> FP32保存下来
            # 所以构建bucket_to_optim_param_groups_map的过程与optimizer的param_groups重新构建放在一块
            self.bucket_to_optim_param_groups_map = {}
            # 同时构建bucket_to_model_params_map，bucket_to_main_params_map
            self.bucket_to_model_params_map = {}
            self.bucket_to_main_params_map = {}
                
            # overlap_optim_step需要使用wrapper包裹原hook
            # 基于记录好的cur_comm_hook, cur_comm_hook_args, cur_comm_hook_kwargs来重新构建hook
            # 注意，这里传的是包裹后的optimizer，即self
            for model_chunk in model_chunks:
                model_chunk.register_comm_hook(stream_wrapper(overlap_optim_step_wrapper(model_chunk.cur_comm_hook, self)),
                                               *model_chunk.cur_comm_hook_args, **model_chunk.cur_comm_hook_kwargs)

        # Handle main parameters.
        # Three groups of parameters:
        #   float16_groups: original float16 parameters，原始模型的fp16参数
        #   fp32_from_float16_groups: fp32 copy of float16 parameters，将原始模型fp16参数转换为optim中的fp32参数
        #   fp32_from_fp32_groups: original fp32 parameters，原始模型的fp32参数，与optim的fp32参数是共享的，不需要拷贝
        self.float16_groups = []
        self.fp32_from_float16_groups = []
        self.fp32_from_fp32_groups = []

        # For all the groups in the original optimizer:
        # 同时在overlap_optim_step下，构建bucket_to_optim_param_groups_map
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, model_param in enumerate(param_group['params']):
                # 跳过不需要优化的参数
                if not model_param.requires_grad:
                    continue

                # float16 params
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    float16_params_this_group.append(model_param)
                    # 创建fp32的拷贝
                    main_param = model_param.detach().clone().float()
                    # 将参数的属性复制下，这个shared表示共享的参数，如ebedding和最后的layer层
                    if hasattr(model_param, 'shared'):
                        main_param.shared = model_param.shared
                        
                    # 将optim中要优化的参数换为fp32的，并保存起来
                    param_group['params'][i] = main_param
                    fp32_from_float16_params_this_group.append(main_param)
                    
                    # Reset existing state dict key to the new main param.
                    # 重新映射下参数的状态
                    if model_param in self.optimizer.state:
                        self.optimizer.state[main_param] = self.optimizer.state.pop(model_param)
                # fp32 params
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    # 模型参数本身就是fp32的，直接引用即可
                    main_param = model_param
                    fp32_params_this_group.append(main_param)
                    param_group['params'][i] = main_param

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

                # overlap_optim_step下，才会构建bucket_to_optim_param_groups_map
                if self.optim_config.overlap_optim_step:
                    # bucket和optim param_groups的映射关系
                    bucket = self.model_param_to_bucket_map[model_param]
                    if bucket not in self.bucket_to_optim_param_groups_map:
                        # 初始化映射
                        self.bucket_to_optim_param_groups_map[bucket] = [[] for _ in range(len(self.optimizer.param_groups))]
                    # 将optim参数加入到bucket映射的对应的组中
                    self.bucket_to_optim_param_groups_map[bucket][group_idx].append(main_param)
                    
                    # 同时构建bucket_to_model_params_map，bucket_to_main_params_map
                    if bucket not in self.bucket_to_model_params_map:
                        self.bucket_to_model_params_map[bucket] = []
                        self.bucket_to_main_params_map[bucket] = []
                    self.bucket_to_model_params_map[bucket].append(model_param)
                    self.bucket_to_main_params_map[bucket].append(main_param)

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)
        
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self._post_init()
        
        # 保存optim的完整的params_group，用于恢复（overlap_optim_step下需要恢复）
        self.all_optim_param_groups = [[param for param in group["params"]] for group in self.optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        """
        We only need to zero the model related parameters, i.e.,
        float16_groups & fp32_from_fp32_groups. We additionally zero
        fp32_from_float16_groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.
        
        只需要将模型相关的参数置零，即float16_groups和fp32_from_fp32_groups
        因为后续会把model中grads的值复制到optim相应参数的grad中。
        
        这里对optim的grad进行zero，是为了减少内存碎片，同时减少内存峰值
        
        1. 清零DDP中grad的buffer
        2. 清空optim中所有参数的grad
        """
        # overlap_optim_step下，会在comm_hook中清零对应的bucket的grad_buffer，尽量实现重叠
        # 1. 清零DDP中grad的buffer
        # 将model中的初始化放进来，方便用户像原PyTorch一样使用
        # 作用：因为DDP中自己管理grad的buffer，需要在每次forward前清零
        # self.optim_config.overlap_optim_step and self.optim_config.overlap_zero_grad_buffer下才会按照bucket进行zero_grad_buffer
        for model_chunk in self.model_chunks:
            model_chunk.zero_grad_buffer(zero_grad_data=not self.optim_config.overlap_optim_step or not self.optim_config.overlap_zero_grad_buffer)

        # 2. 清空全部param的grad，这个不需要分bucket进行zero，因为只是设置为None，不需要启动kernel
        for group in self.float16_groups:
            zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            zero_grad_group_helper(group, set_to_none)
            
        self.reset()  # 记得清零，实现在Z2allOptimizer中

    def _get_model_and_main_params_data_float16(self, is_bucket_step=False, bucket: Bucket = None):
        """
        获取model和optim对应的fp16的参数
        """
        model_data = []
        main_data = []
        
        if not is_bucket_step:
            for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
                for model_param, main_param in zip(model_group, main_group):
                    model_data.append(model_param.data)
                    main_data.append(main_param.data)
        else:
            for model_param, main_param in zip(self.bucket_to_model_params_map[bucket], self.bucket_to_main_params_map[bucket]):
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    model_data.append(model_param.data)
                    main_data.append(main_param.data)
        
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self, is_bucket_step=False, bucket: Bucket = None):
        """
        用于更新放在optim中的主参数，optim main params
        
        将模型的梯度拷贝到优化器的梯度中，优化器中的梯度保存的是fp32的梯度
        
        overlap_optim_step=True, is_bucket_step=True，只需要拷贝对应的bucket
        """
        if not is_bucket_step:
            # 外部有条件：
            # (not self.optim_config.overlap_optim_step) or (self.optim_config.overlap_optim_step and is_bucket_step)
            
            # This only needs to be done for the float16 group.
            for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
                for model_param, main_param in zip(model_group, main_group):
                    if hasattr(model_param, 'main_grad'):
                        main_param.grad = model_param.main_grad.float()
                    else:
                        if model_param.grad is not None:
                            main_param.grad = model_param.grad.float()
                    if not self.optim_config.grad_scaling_before_comm:
                        main_param.grad.mul_(self.optim_config.grad_scaling_factor)

                    # Safe to deallocate model's grad/main_grad after copying.
                    # (If using contiguous buffers, main_grad's memory should
                    # persist and therefore should not be deallocated.)
                    model_param.grad = None

            # For fp32 grads, we need to reset the grads to main grad.
            for model_group in self.fp32_from_fp32_groups:
                for model_param in model_group:
                    model_param.grad = model_param.main_grad
        else:
            # overlap_optim_step下，只对bucket对应的grad进行拷贝
            # 对FP32和FP16都是一样的操作，如果是FP32，则float()也不会创建新的内存空间，所以可以这样做
            for model_param, main_param in zip(self.bucket_to_model_params_map[bucket], self.bucket_to_main_params_map[bucket]):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()
                if not self.optim_config.grad_scaling_before_comm:
                    main_param.grad.mul_(self.optim_config.grad_scaling_factor)
                        
                model_param.grad = None
    
    def _copy_main_params_to_model_params(self, is_bucket_step=False, bucket: Bucket = None):
        """
        用于模型forward和backward
        
        因为模型执行时用的是模型的参数，优化器的参数是单独存放的，fp32的参数是共享的
        
        将优化器的参数拷贝到模型的参数中，优化器中的参数保存的是fp32的参数
        
        overlap_optim_step=True, is_bucket_step=True，只需要拷贝对应的bucket
        """
        # 外部有条件：
        # (not self.optim_config.overlap_optim_step) or (self.optim_config.overlap_optim_step and is_bucket_step)
            
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16(is_bucket_step, bucket)
        _multi_tensor_copy_this_to_that(
            this=main_data, that=model_data, overflow_buf=self._dummy_overflow_buf
        )

    # 没用到
    # def _copy_model_params_to_main_params(self):
    #     # Only needed for the float16 params.
    #     model_data, main_data = self._get_model_and_main_params_data_float16()
    #     _multi_tensor_copy_this_to_that(
    #         this=model_data, that=main_data, overflow_buf=self._dummy_overflow_buf
    #     )

    def state_dict(self):
        state_dict = {}
        state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['fp32_from_fp16_params'] = self.fp32_from_float16_groups
        return state_dict

    def load_state_dict(self, state_dict):
        # Optimizer.
        self.optimizer.load_state_dict(state_dict['optimizer'])

        # Copy data for the main params.
        for current_group, saved_group in zip(
            self.fp32_from_float16_groups, state_dict['fp32_from_fp16_params']
        ):
            for current_param, saved_param in zip(current_group, saved_group):
                current_param.data.copy_(saved_param.data)


class FP32Optimizer(Z2allOptimizer):
    """FP32 Optimizer.
    
    简单的包装，添加将main_grad拷贝到grad的处理
    
    要与DDP配合使用
    
    overlap_optim_step要用到model
    
    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        optim_config (OptimizerConfig): configuration object for optimizer.
        model_chunks: ddp model list.
        scaler: used for scaling gradients.
        grad_clip: used for cliping gradients.
    """
    
    @classmethod
    def _build_bucket_to_optim_param_groups_map(cls, optimizer, model_chunks):
        """
        构建bucket到optim_params的映射，用于overlap_optim_step
        """
        # overlap_optim_step需要实现self.bucket_to_optim_param_groups_map
        bucket_to_optim_param_groups_map = {}
        bucket_to_model_params_map = {}
        bucket_to_main_params_map = {}
            
        # 1. 首先构建原model param到bucket的映射
        # 同时构建bucket_to_model_params_map，bucket_to_main_params_map
        model_param_to_bucket_map = {}
        for model_chunk in model_chunks:
            for buffer in model_chunk.buffers:
                for bucket in buffer.buckets:
                    for param in bucket.params:
                        model_param_to_bucket_map[param] = bucket
                        if bucket not in bucket_to_model_params_map:
                            bucket_to_model_params_map[bucket] = []
                        bucket_to_model_params_map[bucket].append(param)
        bucket_to_main_params_map = bucket_to_model_params_map
        
        # 2. 构建bucket_to_optim_param_groups_map，遍历optim组
        for group_idx, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                bucket = model_param_to_bucket_map[param]
                if bucket not in bucket_to_optim_param_groups_map:
                    # 初始化映射
                    bucket_to_optim_param_groups_map[bucket] = [[] for _ in range(len(optimizer.param_groups))]
                # 将shard的参数加入到对应的组中
                bucket_to_optim_param_groups_map[bucket][group_idx].append(param)
                
        return bucket_to_optim_param_groups_map, bucket_to_model_params_map, bucket_to_main_params_map

    def __init__(self, optimizer: torch.optim.Optimizer,
                 optim_config: OptimizerConfig,
                 model_chunks: Optional[Union[DistributedDataParallel, List[DistributedDataParallel]]],
                 scaler: torch.cuda.amp.GradScaler = None, grad_clip = 0.0):
        """
        简单的Optimizer包裹，需要和MyDDP配合使用
        
        model_chunks可能有多个model_chunk，虚拟流水线的情况下，每个model_chunk对应一个stage
        
        model_chunks必须传递进来，用于合并下面代码，方便用户像原PyTorch一样使用：
        model.zero_grad_buffer()
        optimizer.zero_grad(set_to_none=True)
        

        overlap_optim_step需要实现：
        1. `optimizer.bucket_to_optim_param_groups_map`, 用于optimizer.step()

        2. `optimizer.bucket_to_model_params_map`, `optimizer.bucket_to_main_params_map`, 两个目的：

        * `main_param.grad = model_param.main_grad.float()`，复制梯度，用于optim参数更新
        * `model_param.copy_(main_param)`，将更新后的optim参数复制回model
        """
        
        super().__init__(optimizer, optim_config, scaler, grad_clip)
        
        assert model_chunks is not None, "model is required!"
        if not isinstance(model_chunks, List):
            model_chunks = [model_chunks]
        self.model_chunks = model_chunks
        
        # 梯度通信后，立即进行参数更新
        if self.optim_config.overlap_optim_step:
            # 这个重叠必须要有overlap_grad_reduce，要不然没意义了
            for model_chunk in self.model_chunks:
                assert model_chunk.ddp_config.overlap_grad_reduce, "ddp_config.overlap_grad_reduce is required for overlap_optim_step"
                
            # overlap_optim_step需要构建self.bucket_to_optim_param_groups_map
            # 同时使用overlap_optim_step_wrapper包裹原hook
            # self.bucket_to_optim_param_groups_map = {bucket: [group_params, ...]}
            (
                self.bucket_to_optim_param_groups_map,
                self.bucket_to_model_params_map,
                self.bucket_to_main_params_map
            ) = self._build_bucket_to_optim_param_groups_map(optimizer, self.model_chunks)
                
            # overlap_optim_step需要使用wrapper包裹原hook
            # 基于记录好的cur_comm_hook, cur_comm_hook_args, cur_comm_hook_kwargs来重新构建hook
            # 注意，这里传的是包裹后的optimizer，即self
            for model_chunk in self.model_chunks:
                model_chunk.register_comm_hook(stream_wrapper(overlap_optim_step_wrapper(model_chunk.cur_comm_hook, self)),
                                               *model_chunk.cur_comm_hook_args, **model_chunk.cur_comm_hook_kwargs)
            
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self._post_init()
        
        # 保存optim的完整的params_group，用于恢复（overlap_optim_step下需要恢复）
        self.all_optim_param_groups = [[param for param in group["params"]] for group in self.optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        """
        1. 清零DDP中grad的buffer
        2. 清空optim中所有参数的grad
        """
        # overlap_optim_step下，会在comm_hook中清零对应的bucket的grad_buffer，尽量实现重叠
        # 1. 清零DDP中grad的buffer
        # 将model中的初始化放进来，方便用户像原PyTorch一样使用
        # 作用：因为DDP中自己管理grad的buffer，需要在每次forward前清零
        # self.optim_config.overlap_optim_step and self.optim_config.overlap_zero_grad_buffer下才会按照bucket进行zero_grad_buffer
        for model_chunk in self.model_chunks:
            model_chunk.zero_grad_buffer(zero_grad_data=not (self.optim_config.overlap_optim_step and self.optim_config.overlap_zero_grad_buffer))

        # 2. 清空全部param的grad，这个不需要分bucket进行zero，因为只是设置为None，不需要启动kernel
        # 注意不能对self.optimizer.param_groups中的参数进行zero_grad，因为这里面的参数已经改为bucket了
        zero_grad_group_helper(self.all_params_for_unscaling_grads, set_to_none)
        
        self.reset()  # 记得清零，实现在Z2allOptimizer中
    
    def _copy_model_main_grads_to_grads(self, is_bucket_step=False, bucket: Bucket = None):
        """
        Copy model main_grad to grad.
        1. 因为累加时是对main_grad进行的，所以这里需要将main_grad拷贝到grad
        2. 为什么要对main_grad累加，而不是直接对grad
          因为对grad进行了内存管理（放在main_grad中），来减少内存碎片
          
        这里会有两种行为：
        1. overlap_optim_step=False，拷贝全部参数
        2. overlap_optim_step=True, is_bucket_step=True，只需要拷贝对应的bucket
        """
        if not is_bucket_step:
            # 外部有条件：
            # (not self.optim_config.overlap_optim_step) or (self.optim_config.overlap_optim_step and is_bucket_step)
            
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad = param.main_grad
                    if not self.optim_config.grad_scaling_before_comm:
                        param.grad.mul_(self.optim_config.grad_scaling_factor)
        else:
            for param_idx in range(len(self.bucket_to_model_params_map[bucket])):
                self.bucket_to_main_params_map[bucket][param_idx].grad = self.bucket_to_model_params_map[bucket][param_idx].main_grad
                if not self.optim_config.grad_scaling_before_comm:
                    self.bucket_to_main_params_map[bucket][param_idx].grad.mul_(self.optim_config.grad_scaling_factor)

    @torch.no_grad()
    def step(self, is_bucket_step=False, bucket: Bucket = None):
        """
        is_bucket_step, bucket用于overlap_optim_step
        
        1. 若overlap_optim_step=False，等待梯度同步完成，然后才能进行更新
        
        2. 更新参数，包括复制梯度到优化器的梯度中，更新参数。
            非overlap_optim_step，直接更新所有参数
            overlap_optim_step，只有当is_bucket_step=True时，才会触发更新（在overlap_optim_step_wrapper包裹的comm_hook中进行更新）
            
        3. 若overlap_optim_step=True and not is_bucket_step，则等待参数更新完成
            是全部参数，进行等待，即在comm_hook中不会等，只会所有触发后才会等
        """
        if not self.optim_config.overlap_optim_step:
            # 1. 若overlap_optim_step=False，等待梯度同步完成
            for model_chunk in self.model_chunks:
                model_chunk.finish_grad_sync()
                
        if (not self.optim_config.overlap_optim_step) or (self.optim_config.overlap_optim_step and is_bucket_step):
            # Copy model main_grad to grad.
            self._copy_model_main_grads_to_grads(is_bucket_step, bucket)
            
            # FP32不需要scaler
            # 2. 裁剪梯度，更新参数
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self._collect_main_params_for_unscaling_grads(), self.grad_clip)
            self.optimizer.step()
        
        if self.optim_config.overlap_optim_step and not is_bucket_step:
            # 3. 等待参数更新完成
            for model_chunk in self.model_chunks:
                model_chunk.finish_grad_sync()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
