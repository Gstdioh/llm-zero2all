from typing import List, Optional, Union
from abc import ABC, abstractmethod

import torch
import amp_C
from apex.multi_tensor_apply import multi_tensor_applier

from optimizer import OptimizerConfig
from parallel.distributed_data_parallel import DistributedDataParallel
from parallel.distributed_data_parallel.ddp_comm_hooks.overlap_optim_step_hooks import overlap_optim_step_wrapper


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
    if overflow_buf is not None:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(amp_C.multi_tensor_scale, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


class Z2allOptimizer:
    """
    简单的包裹，使得访问Z2allOptimizer就像访问self.optimizer一样
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, optim_config: OptimizerConfig,
                 scaler: torch.cuda.amp.GradScaler = None, grad_clip=0.0):
        self.optimizer = optimizer
        self.optim_config = optim_config
        self.scaler = scaler
        self.grad_clip = grad_clip
        
    def _post_init(self):
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self.all_params_for_unscaling_grads = [
            param for group in self.optimizer.param_groups for param in group["params"] 
            if param.requires_grad and param.grad is not None
        ]
        
    def __getattr__(self, name):
        """
        当访问NavieOptimizer的属性时，如果没有找到，就去optimizer中找
        
        相当于将NavieOptimizer视为optimizer的一个包装，对外部代码无影响
        """
        return getattr(self.optimizer, name)


class MixedPrecisionOptimizer(Z2allOptimizer, ABC):
    """Base class for both the float-16 and the distributed optimizer.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
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
    def step(self):
        """
        更新参数，若需要则可以进行scale和grad_clip
        
        Return:
        update_successful (bool): True if the update was successful.
            若梯度中包含Nan，则不进行更新，逻辑在scaler.step中
        """
        # Copy gradients from model params to main params.
        self._copy_model_grads_to_main_grads()
        
        update_successful = True
        
        if self.scaler is not None and self.scaler.is_enabled():
            # unscale，并且裁剪梯度
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.all_params_for_unscaling_grads, self.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)  # 若没有unscale，进行unscale并且更新
            
            # 判断是否更新成功
            optimizer_state = self.scaler._per_optimizer_states[id(self.optimizer)]
            if sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
                # 有Nan或者inf
                update_successful = False
            
            self.scaler.update()  # 更新scaler的缩放因子，为了进行下一个iter，会清空scale中optim的状态
        else:
            # 裁剪梯度
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.all_params_for_unscaling_grads, self.grad_clip)
            self.optimizer.step()

        # Update params from optim params.
        # 更新失败的话，就不用覆盖模型参数了
        if update_successful:
            self._copy_main_params_to_model_params()
            
        return update_successful


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Float16 optimizer for fp16 and bf16 data types.

    Args:
        optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
        config (OptimizerConfig): configuration object for optimizer.
        grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
            this can be None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constant gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        init_state_fn (Callable, optional): function to initialize state in the optimizer.
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 optim_config: OptimizerConfig, scaler: torch.cuda.amp.GradScaler = None, grad_clip = 0.0,
                 model_chunks: Optional[Union[DistributedDataParallel, List[DistributedDataParallel]]] = None):
        super().__init__(optimizer, optim_config, scaler, grad_clip)

        # overlap_optim_step需要实现self.bucket_to_optim_params_map
        if self.optim_config.overlap_optim_step:
            assert model_chunks is not None, "model is required for overlap_optim_step"
            if not isinstance(model_chunks, List):
                model_chunks = [model_chunks]
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
                            
            # 2. 构建bucket_to_optim_params_map，遍历optim组
            # 代码逻辑与FP32的不太一样，因为optimizer的param_groups会重新构建，即将FP16 -> FP32保存下来
            # 所以构建bucket_to_optim_params_map的过程与optimizer的param_groups重新构建放在一块
            self.bucket_to_optim_params_map = {}
                
            # overlap_optim_step需要使用wrapper包裹原hook
            # 基于记录好的cur_comm_hook, cur_comm_hook_args, cur_comm_hook_kwargs来重新构建hook
            # 注意，这里传的是包裹后的optimizer，即self
            for model_chunk in model_chunks:
                model_chunk.register_comm_hook(overlap_optim_step_wrapper(model_chunk.cur_comm_hook, self),
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
        # 同时在overlap_optim_step下，构建bucket_to_optim_params_map
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            float16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_float16_params_this_group = []
            # For all the parameters in this group:
            for i, model_param in enumerate(param_group['params']):
                if model_param.requires_grad:

                    # float16 params:
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
                    # fp32 params.
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

                # overlap_optim_step下，才会构建bucket_to_optim_params_map
                if self.optim_config.overlap_optim_step:
                    # bucket和optim param_groups的映射关系
                    bucket = self.model_param_to_bucket_map[model_param]
                    if bucket not in self.bucket_to_optim_params_map:
                        # 初始化映射
                        self.bucket_to_optim_params_map[bucket] = [[] for _ in range(len(self.optimizer.param_groups))]
                    # 将optim参数加入到bucket映射的对应的组中
                    self.bucket_to_optim_params_map[bucket][group_idx].append(main_param)

            self.float16_groups.append(float16_params_this_group)
            self.fp32_from_float16_groups.append(fp32_from_float16_params_this_group)
            self.fp32_from_fp32_groups.append(fp32_params_this_group)
        
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self._post_init()

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
        """
        for group in self.float16_groups:
            zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_float16_groups:
            zero_grad_group_helper(group, set_to_none)
        for group in self.fp32_from_fp32_groups:
            zero_grad_group_helper(group, set_to_none)

    def _get_model_and_main_params_data_float16(self):
        """
        获取model和optim对应的fp16的参数
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """
        用于更新放在optim中的主参数，optim main params
        
        将模型的梯度拷贝到优化器的梯度中，优化器中的梯度保存的是fp32的梯度
        """
        # This only needs to be done for the float16 group.
        for model_group, main_group in zip(self.float16_groups, self.fp32_from_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                if hasattr(model_param, 'main_grad'):
                    main_param.grad = model_param.main_grad.float()
                else:
                    if model_param.grad is not None:
                        main_param.grad = model_param.grad.float()

                # Safe to deallocate model's grad/main_grad after copying.
                # (If using contiguous buffers, main_grad's memory should
                # persist and therefore should not be deallocated.)
                model_param.grad = None

        # For fp32 grads, we need to reset the grads to main grad.
        for model_group in self.fp32_from_fp32_groups:
            for model_param in model_group:
                model_param.grad = model_param.main_grad
    
    def _copy_main_params_to_model_params(self):
        """
        用于模型forward和backward
        
        因为模型执行时用的是模型的参数，优化器的参数是单独存放的，fp32的参数是共享的
        
        将优化器的参数拷贝到模型的参数中，优化器中的参数保存的是fp32的参数
        """
        # Only needed for the float16 params.
        model_data, main_data = self._get_model_and_main_params_data_float16()
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
    """
    
    @classmethod
    def _build_bucket_to_optim_params_map(cls, optimizer, model_chunks):
        """
        构建bucket到optim_params的映射，用于overlap_optim_step
        """
        # overlap_optim_step需要实现self.bucket_to_optim_params_map
        bucket_to_optim_params_map = {}
            
        # 1. 首先构建原model param到bucket的映射
        model_param_to_bucket_map = {}
        for model_chunk in model_chunks:
            for buffer in model_chunk.buffers:
                for bucket in buffer.buckets:
                    for param in bucket.params:
                        model_param_to_bucket_map[param] = bucket
        
        # 2. 构建bucket_to_optim_params_map，遍历optim组
        for group_idx, param_group in enumerate(optimizer.param_groups):
            for param in param_group['params']:
                bucket = model_param_to_bucket_map[param]
                if bucket not in bucket_to_optim_params_map:
                    # 初始化映射
                    bucket_to_optim_params_map[bucket] = [[] for _ in range(len(optimizer.param_groups))]
                # 将shard的参数加入到对应的组中
                bucket_to_optim_params_map[bucket][group_idx].append(param)
                
        return model_param_to_bucket_map, bucket_to_optim_params_map

    def __init__(self, optimizer: torch.optim.Optimizer,
                 optim_config: OptimizerConfig, scaler: torch.cuda.amp.GradScaler = None, grad_clip = 0.0,
                 model_chunks: Optional[Union[DistributedDataParallel, List[DistributedDataParallel]]] = None):
        """
        model_chunks可能有多个model_chunk，虚拟流水线的情况下，每个model_chunk对应一个stage
        """
        
        super().__init__(optimizer, optim_config, scaler, grad_clip)
        
        # 梯度通信后，立即进行参数更新
        if self.optim_config.overlap_optim_step:
            assert model_chunks is not None, "model is required for overlap_optim_step"
            if not isinstance(model_chunks, List):
                model_chunks = [model_chunks]
            # 这个重叠必须要有overlap_grad_reduce，要不然没意义了
            for model_chunk in model_chunks:
                assert model_chunk.ddp_config.overlap_grad_reduce, "ddp_config.overlap_grad_reduce is required for overlap_optim_step"
                
            # overlap_optim_step需要构建self.bucket_to_optim_params_map
            # 同时使用overlap_optim_step_wrapper包裹原hook
            # self.bucket_to_optim_params_map = {bucket: [group_params, ...]}
            (
                self.model_param_to_bucket_map,
                self.bucket_to_optim_params_map
            ) = self._build_bucket_to_optim_params_map(optimizer, model_chunks)
                
            # overlap_optim_step需要使用wrapper包裹原hook
            # 基于记录好的cur_comm_hook, cur_comm_hook_args, cur_comm_hook_kwargs来重新构建hook
            # 注意，这里传的是包裹后的optimizer，即self
            for model_chunk in model_chunks:
                model_chunk.register_comm_hook(overlap_optim_step_wrapper(model_chunk.cur_comm_hook, self),
                                               *model_chunk.cur_comm_hook_args, **model_chunk.cur_comm_hook_kwargs)
            
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self._post_init()

    def zero_grad(self, set_to_none=True):
        """Copied from torch.optim.optimizer"""
        for group in self.optimizer.param_groups:
            zero_grad_group_helper(group['params'], set_to_none)
    
    def _copy_model_main_grads_to_grads(self):
        """
        Copy model main_grad to grad.
        1. 因为累加时是对main_grad进行的，所以这里需要将main_grad拷贝到grad
        2. 为什么要对main_grad累加，而不是直接对grad
          因为对grad进行了内存管理（放在main_grad中），来减少内存碎片
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = param.main_grad

    @torch.no_grad()
    def step(self):
        # Copy model main_grad to grad.
        self._copy_model_main_grads_to_grads()
        
        if self.scaler is not None:
            # unscale，并且裁剪梯度
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.all_params_for_unscaling_grads, self.grad_clip)
            # step the optimizer and scaler if training in fp16
            self.scaler.step(self.optimizer)  # 若没有unscale，进行unscale并且更新
            self.scaler.update()  # 更新scaler的缩放因子
        else:
            # 裁剪梯度
            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.all_params_for_unscaling_grads, self.grad_clip)
            self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
