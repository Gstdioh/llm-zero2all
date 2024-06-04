# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import os
from enum import Enum
from logging import getLogger
from typing import Dict, List, Optional

import torch
import torch.distributed

from .distributed_data_parallel_config import DistributedDataParallelConfig


logger = getLogger(__name__)


class BufferType(Enum):
    PARAM = 1
    GRAD = 2


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Args:
        ddp_config: DistributedDataParallel config object.
        params: List of parameters whose gradients are collated in this bucket.
        param_data: View in larger ParamAndGradBuffer.param_data that this bucket is responsible for.
        grad_data: View in larger ParamAndGradBuffer.grad_data that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger ParamAndGradBuffer.
        numel_unpadded: Number of unpadded elements in bucket.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],
        grad_data: torch.Tensor,
        offset: int,
        numel_unpadded: int,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        gbuf_index: int,
        local_bucket_id: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
    ):
        self.ddp_config = ddp_config

        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        # 当这个bucker的所有参数都计算出了梯度，就会发起通信操作，计算和通信重叠
        self.params_list = params  # 有序
        self.params = set(params)  # 无序，和params_with_grad结合使用判断param是否都准备好了
        self.params_with_grad = set()  # 记录准备好的参数，全部准备好后发起通信
        self.param_data = param_data
        self.grad_data = grad_data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.numel_unpadded = numel_unpadded
        
        self.raw_grad_data = None  # 用于保存原始的梯度数据，用于恢复
        
        # 存储每个param的bucket下的数据范围，(start_index, end_index)
        local_offset = 0
        self.param_index_map = {}
        for param in self.params_list:
            start_index, end_index = local_offset, local_offset + param.data.nelement()
            self.param_index_map[param] = (start_index, end_index)
            local_offset += param.data.nelement()
        
        # 唯一标识
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.gbuf_index = gbuf_index  # buffer的全局id
        self.global_bucket_id = None  # int, bucket的全局id，在DDP设置完buffer后，会设置该值，倒序
        self.global_n_buckets = None  # int, DDP下所有bucket的数量，在DDP设置完buffer后，会设置该值
        self.local_bucket_id = local_bucket_id  # bucket在buffer中的id，倒序
        self.local_n_buckets = None  # int, buffer中bucket的数量，buffer设置完所有bucket后，会设置该值
        
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        
        self.comm_call = None  # 执行bucket通信的函数
        
        self.reset()

    def index(self) -> int:
        return self.global_bucket_id
    
    def buffer(self) -> torch.Tensor:
        """
        bucket grad buffer
        为了和pytorch兼容，所以没有命名为grad_buffer
        """
        return self.grad_data
    
    def set_buffer(self, tensor: torch.Tensor) -> None:
        """
        修改bucket的grad buffer的值，不会改变buffer的数据类型
        
        tensor和buffer的元素总数一样即可（不要求形状相同）
        """
        if self.grad_data.dtype != tensor.dtype:
            logger.warning("注意，Bucket.set_buffer(tensor)只能修改buffer的值（不会改变buffer的数据类型），\
                            若要修改buffer的数据类型（如fp16_compress_wrapper_hook下），\
                            请使用change_grad_buffer_dtype(new_grad_type)，记得要恢复")
        
        self.grad_data.copy_(tensor)
    
    def param_buffer(self) -> torch.Tensor:
        """
        bucket param buffer
        """
        return self.param_data
    
    def set_param_buffer(self, tensor: torch.Tensor) -> None:
        """
        修改bucket的param buffer的值，不会改变buffer的数据类型
        
        tensor和buffer的元素总数一样即可（不要求形状相同）
        """
        self.param_data.copy_(tensor)
        
    def gradients(self) -> List[torch.Tensor]:
        """
        grad list
        """
        grad_list = []
        
        for param in self.params_list:
            start_index, end_index = self.param_index_map[param]
            grad_list.append(self.grad_data[start_index: end_index].view(param.shape))
        
        return grad_list
        
    def parameters(self) -> List[torch.Tensor]:
        """
        param list
        """
        return self.params_list
        
    def is_last(self) -> bool:
        """
        是否是全局bucket下的最后一个bucket
        
        因为buffer是按正序，bucket是按倒序，所以这样判断
        """
        return self.global_bucket_id == self.global_n_buckets - 1
    
    def change_grad_buffer_dtype(self, new_grad_dtype) -> None:
        """
        修改grad buffer的数据类型，需要保存之前的数据，后面必须使用restore_grad_buffer_dtype()恢复
        """
        self.raw_grad_data = self.grad_data
        self.grad_data = self.grad_data.to(new_grad_dtype)
        
    def restore_grad_buffer_dtype(self) -> None:
        """
        恢复grad buffer的数据类型
        """
        self.raw_grad_data.copy_(self.grad_data)
        self.grad_data = self.raw_grad_data
        self.raw_grad_data = None

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False
        
    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        # Make sure norm of grads in bucket are not NaN
        # prior to data-parallel all-reduce / reduce-scatter.
        # torch的scaler会自动处理NaN，发现有NaN就会不更新参数，并且修改scale值
        if self.ddp_config.check_for_nan_in_grad:
            global_rank = torch.distributed.get_rank()
            norm = self.grad_data.norm(p=2)
            assert not norm.isnan(), (
                f'Rank {global_rank}: found NaN in local grad norm in '
                f'backward pass before data-parallel communication collective. '
                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
            )
           
        self.communication_handle = self.comm_call()  # 触发通信操作，返回通信句柄（有wait()方法）
        self.communication_issued = True

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.ddp_config.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()
    
    def register_comm_hook(self, hook: callable, *args, **kwargs):
        """
        Registers a communication hook for the bucket.
        
        同时返回当前的comm_hook，用于后续对该comm_hook的修改
        """
        def comm_call():
            return hook(self, *args, **kwargs)
            
        self.comm_call = comm_call


class ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        gbuf_index: int,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
    ):
        self.ddp_config = ddp_config
        
        self.gbuf_index = gbuf_index

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.is_last_microbatch = True  # 梯度累加时，只有最后一个microbatch才会同步梯度

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            因为分布式优化器会对数据进行均匀划分，用于reduce-scatter操作
            """
            if self.ddp_config.use_distributed_optimizer:
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                return _pad(data_index, math.lcm(self.data_parallel_world_size, 128))
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []  # 保存的范围是pad的
        per_bucket_numel_unpadded = []
        bucket_id = 0

        def _create_new_bucket(data_end_index: int) -> int:
            """
            Create the bucket_id'th bucket with collected bucket_params, starting at
            bucket_data_start_index.
            """
            nonlocal bucket_data_start_index, bucket_params, bucket_id
            per_bucket_numel_unpadded.append(data_end_index - bucket_data_start_index)
            data_end_index = _pad_if_needed(data_end_index)
            # Update bucket metadata.
            self.bucket_indices.append((bucket_data_start_index, data_end_index))
            bucket_data_start_index = data_end_index
            # Re-set bucket_params and increment bucket_id for next bucket.
            bucket_params = set()
            bucket_id += 1
            # Return the potentially padded data_end_index.
            return data_end_index

        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel

            def _does_param_require_new_bucket(param):
                """
                将共享的嵌入参数分割到单独的bucket中
                Split shared embedding parameters into separate bucket if using distributed
                optimizer that makes use of reduce-scatters instead of all-reduces.
                This ensures that the first and last pipeline stage partition optimizer state
                for the shared embedding parameters the same way across DP replicas, allowing
                the DP reduce-scatter to be before the embedding all-reduce.
                确保在第一阶段和最后阶段的pipeline中，对共享嵌入参数的优化器状态的划分在所有的DP副本中是一致的
                """
                return (
                    getattr(param, "shared_embedding", False)
                    and self.ddp_config.use_distributed_optimizer
                )

            # Create bucket with already collected parameters if current param needs its own bucket.
            if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
                # We are creating a bucket for the already accumulated parameters, whose params
                # end at the current data_start_index.
                if self.ddp_config.use_distributed_optimizer:
                    # data_start_index should already be padded.
                    assert data_start_index % self.data_parallel_world_size == 0
                _create_new_bucket(data_start_index)

            # 保存的是真实param大小的起始和末尾，没有pad
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already or the current param is part of the shared embedding
            # layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None
                and (data_end_index - bucket_data_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                data_end_index = _create_new_bucket(data_end_index)
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            data_end_index = _create_new_bucket(data_end_index)

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        else:
            assert self.numel == self.numel_unpadded

        self.param_data = None
        # Only re-map param tensors if using distributed optimizer.
        # 分布式优化器下，才需要对模型参数进行映射
        if self.ddp_config.use_distributed_optimizer:
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(
            self.numel,
            dtype=self.grad_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # Finally, map param.data and param.main_grad fields to buffers.
        bucket_params = []
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]

            # Assign param.data to appropriate segment of self.param_data.
            if self.param_data is not None:
                old_param_data = param.data
                param.data = self._get(
                    param.data.shape, data_start_index, buffer_type=BufferType.PARAM
                )
                # 在 PyTorch 中，_base 属性是一个内部属性，
                # 用于跟踪一个张量（Tensor）是否是另一个张量的视图（View）。
                # 如果一个张量是另一个张量的视图，那么这个张量的 _base 属性就会指向原始张量。
                # 如果一个张量不是任何其他张量的视图，那么它的 _base 属性就会是 None。
                # 删除原始张量才会释放显存
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data  # 将原来的param.data删除，释放显存

            param.main_grad = self._get(
                param.data.shape, data_start_index, buffer_type=BufferType.GRAD
            )
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_data_start_index,
                    end_index=bucket_data_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = []
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.append(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params=bucket_params,
                start_index=bucket_data_start_index,
                end_index=bucket_data_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )
            
        # 设置完所有bucket后，通知所有bucket，当前buffer中所含bucket的数量
        for bucket in self.buckets:
            bucket.local_n_buckets = len(self.buckets)

        if torch.distributed.get_rank() == 0:
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}')
                    
    def scale_gradients(self, scaling_factor: float) -> None:
        """Scale the gradient data by `scaling_factor`."""
        self.grad_data *= scaling_factor

    def _get(self, shape: torch.Size, start_index: int, buffer_type: BufferType) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        if buffer_type == BufferType.PARAM:
            assert self.param_data is not None
            buffer_tensor = self.param_data[start_index:end_index]
        elif buffer_type == BufferType.GRAD:
            buffer_tensor = self.grad_data[start_index:end_index]
        else:
            raise Exception("Illegal buffer type provided to GradBuffer._get() function")
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        numel_unpadded: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.ddp_config.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global ParamAndGradBuffer.
        bucketed_param_data = None
        if self.param_data is not None:
            bucketed_param_data = self._get(
                torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.PARAM
            )
        bucketed_grad_data = self._get(
            torch.Size([end_index - start_index]), start_index, buffer_type=BufferType.GRAD
        )
        bucket = Bucket(
            ddp_config=self.ddp_config,
            params=bucket_params,
            param_data=bucketed_param_data,
            grad_data=bucketed_grad_data,
            offset=start_index,
            numel_unpadded=numel_unpadded,
            param_dtype=self.param_dtype,
            grad_dtype=self.grad_dtype,
            gbuf_index=self.gbuf_index,
            local_bucket_id=bucket_id,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self, zero_grad_data: bool = True):
        """
        Zero out the underlying grad_buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        if zero_grad_data:
            self.grad_data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.ddp_config.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        # 考虑gradient accumulation，只有在最后一个microbatch时才会进行梯度同步
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)
    
    def register_comm_hook(self, hook: callable, *args, **kwargs):
        """
        Registers a communication hook for all the buckets.
        """
        for bucket in self.buckets:
            bucket.register_comm_hook(hook, *args, **kwargs)
