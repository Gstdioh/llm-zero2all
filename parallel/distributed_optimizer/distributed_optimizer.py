# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


import itertools
from logging import getLogger
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
from apex.optimizers import FusedAdam as Adam

from parallel.distributed_data_parallel import ParamAndGradBuffer, Bucket, DistributedDataParallel
from parallel.distributed_data_parallel.ddp_comm_hooks.overlap_optim_step_hooks import overlap_optim_step_wrapper
from optimizer import OptimizerConfig, MixedPrecisionOptimizer, zero_grad_group_helper


logger = getLogger(__name__)


def shards_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    
    返回分割的列表（用于分布式优化器）
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer


class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0):
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start


class DistributedOptimizer(MixedPrecisionOptimizer):
    @classmethod
    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        gbuf_world_range: Range,
        bucket_offset: int,
    ):
        """
        Build mapping from param reference to grad buffer shard ranges.

        This method builds a mapping from parameter references to grad
        buffer shard ranges, specific to each data-parallel (DP) rank's
        set of 'owned' parameters. Each grad buffer (padded to be an even
        multiple of DP-world-size) is conceptually divided into DP-world-size
        contiguous regions, where each DP rank 'owns' a contiguous regions.
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        
        Input:
            param_world_index_map: Dict[torch.nn.Parameter, Tuple],
                保存了参数在全局梯度缓冲区中的起始和结束下标，以及桶的id。
                例子：
                    {
                        param1: (0, 100, 0),
                        param2: (100, 200, 0),
                        ...
                    }
            gbuf_world_range: Range,
                保存了某个dp rank全局梯度缓冲区的起始和结束下标。
                例子：
                    Range(0, 1000)
            bucket_offset: int,
                桶的偏移量。在全局buffer下的起始下标。
        
        Return:
            返回一个字典，其中键是参数对象，值是另一个字典
            包含了该参数在全局梯度缓冲区、在桶内的全局梯度缓冲区、在本地梯度缓冲区以及在参数自身中的范围。
            
        Example:
            {
                param1: {
                    "gbuf_world": Range(0, 100),  # 全局buffer下的范围
                    "gbuf_world_in_bucket": Range(0, 100),  # bucket下的范围
                    "gbuf_local": Range(0, 100),  # bucket分成dp份，dp rank下的范围
                    "param": Range(0, 100),  # 参数自身的范围，因为参数可能会被分割
                },
                param2: {
                    "gbuf_world": Range(100, 200),
                    "gbuf_world_in_bucket": Range(0, 100),
                    "gbuf_local": Range(0, 100),
                    "param": Range(0, 100),
                },
                ...
            }
        """

        # Param range map.
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():  # 对所有参数分配到相应的dp rank中

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            # 该参数在当前的dp rank范围，则添加
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: ParamAndGradBuffer, bucket_index: int):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        
        Return: 
            对一个桶中的参数
            返回一个字典，其中包含了参数和它们的梯度缓冲区的映射关系。
            {
                "param_map": param_range_map,
            }
            其中，param_map 是一个字典，它的键是参数对象，值是另一个字典，
            包含了该参数在全局梯度缓冲区、在桶内的全局梯度缓冲区、在本地梯度缓冲区以及在参数自身中的范围。

        Example:
            {
                "param_map": {
                    param1: {
                        "gbuf_world": Range(0, 50),
                        "gbuf_world_in_bucket": Range(0, 50),
                        "gbuf_local": Range(0, 50),
                        "param": Range(0, 50),
                    },
                    param2: {
                        "gbuf_world": Range(50, 100),
                        "gbuf_world_in_bucket": Range(0, 50),
                        "gbuf_local": Range(0, 50),
                        "param": Range(0, 50),
                    },
                    ...
                },
            }
        """

        data_parallel_rank = torch.distributed.get_rank(param_and_grad_buffer.data_parallel_group)
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        # 每个dp rank在这个bucket下的全局buffer下标范围
        # 如bucket在buffer的范围是(400, 800)，有2个dp rank，则dp rank0的范围是(400, 600)，dp rank1的范围是(600, 800)
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        """
        param_range_map: {}, 保存的是真实param大小的起始和末尾，没有pad
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
        """
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {
            "param_map": param_range_map,
        }

        return data

    @classmethod
    def _build_gbuf_range_map(cls, param_and_grad_buffer: ParamAndGradBuffer):
        """
        Build mapping between params and their grad buffers. These mappings are
        partitioned according to data type.

        Iterate through all buckets of grad buffer to construct param ranges
        that this rank "owns" (the dp_rank'th shard of each bucket, where each
        shard is 1/dp_world_size of the bucket).

        Args:
            param_and_grad_buffer (ParamAndGradBuffer): buffer to build mapping for.
        
        Return:
            返回一个字典，其中键是参数数据类型和梯度数据类型的元组，值是字典列表。
            列表中的每个字典对应一个桶，包含从参数到其在梯度缓冲区中的范围的映射。
            
        Example:
            假设我们有两个float32类型的参数和它们对应的float16类型的梯度。
            如果我们有两个桶，返回值可能如下所示：
            {
                (torch.float32, torch.float16): [
                    {
                        "param_map": {  # bucket1
                            param1: {
                                "gbuf_world": Range(0, 50),
                                "gbuf_world_in_bucket": Range(0, 50),
                                "gbuf_local": Range(0, 50),
                                "param": Range(0, 50),
                            },
                            param2: {
                                "gbuf_world": Range(50, 100),
                                "gbuf_world_in_bucket": Range(0, 50),
                                "gbuf_local": Range(0, 50),
                                "param": Range(0, 50),
                            },
                            ...
                        },
                    },
                    {
                        "param_map": {  # bucket2
                            param3: {
                                "gbuf_world": Range(100, 200),
                                "gbuf_world_in_bucket": Range(0, 100),
                                "gbuf_local": Range(0, 100),
                                "param": Range(0, 100),
                            },
                        },
                    },
                ]
            }
        """
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }

    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """
        Create a reverse of the gbuf_ranges, for referencing in opposite direction.
        
        Input:
            gbuf_ranges: List[Dict]
            [
                {
                    (torch.float32, torch.float16): [
                        {
                            "param_map": {  # bucket1
                                param1: {
                                    "gbuf_world": Range(0, 50),
                                    "gbuf_world_in_bucket": Range(0, 50),
                                    "gbuf_local": Range(0, 50),
                                    "param": Range(0, 50),
                                },
                                param2: {
                                    "gbuf_world": Range(50, 100),
                                    "gbuf_world_in_bucket": Range(0, 50),
                                    "gbuf_local": Range(0, 50),
                                    "param": Range(0, 50),
                                },
                                ...
                            },
                        },
                        {
                            "param_map": {  # bucket2
                                param3: {
                                    "gbuf_world": Range(100, 200),
                                    "gbuf_world_in_bucket": Range(0, 100),
                                    "gbuf_local": Range(0, 100),
                                    "param": Range(0, 100),
                                },
                            },
                        },
                    ]
                },
                {
                    (torch.float16, torch.float16): [
                        ...
                    ]
                },
                ...
            ]
            
        Return:
            param到具体的梯度缓冲区的映射。
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        assert (
                            param not in param_gbuf_map
                        ), "Param should not be in param_gbuf_map; each param only belongs to a single bucket"
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        
        gbuf_ranges有当前dp rank下所包含的参数的信息（作为key）
        
        Return:
        group_ranges = []
        group_ranges[0] = {
            "params": ,  # 这个group下的参数
            "orig_group": ,  # 原始完整的optim参数组，包含所有dp rank的参数，即self.optimizer.param_groups[group_index]
            "orig_group_idx": ,  # 原始optim参数组的id
        }
        """

        # Param group map.
        # World param group map.
        # - Store a mapping of <model_parameter:group_index> for all parameters
        #   across all DP ranks. This is necessary because it is our first
        #   cross reference between the DDP mappings and the optimizer group
        #   parameters. This mapping only for use in the next step of building
        #   the local mapping over this DP rank's parameters.
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # Optimizer group ranges & param-group mapping.
        # - Build a mapping from groups to their contained parameters, and also
        #   from parameters to their containing group index and order within
        #   the group. The group index and order are particularly important for
        #   saving and loading checkpoints.
        local_param_group_map = {}  # 某个dp rank下的新的param group，参数 -> 所处的组id和组内顺序
        group_ranges = [{"params": []} for _ in param_groups]  # 某个dp rank所包含的参数组，其中的参数还没有进行shard
        for gbuf_range_map in gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]  # 原始完整的optim参数组，包含所有dp rank的参数
            group_range["orig_group_idx"] = group_index

        return local_param_group_map, group_ranges

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
        model_param_to_bucket_map: Dict[torch.nn.Parameter, Bucket],
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        
        同时构建bucket和shard optim param_groups的映射关系，用于overlap_optim_step
        bucket被唯一标识：id(bucket)
        shard optim param_groups： [group_params, ...], optimizer用于优化的参数组，
            元素是某个组的所有参数
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []
        
        # bucket和shard optim param_groups的映射关系
        bucket_to_optim_params_map = {}

        # Allocate (or slice) each group's param shard.
        # opt_group_ranges, 某个dp rank所包含的参数组，其中的参数还没有进行shard
        for group_idx, group_range in enumerate(opt_group_ranges):

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            # 对于所有当前dp组所对应的参数，进行shard
            for model_param in group_range["params"]:

                assert model_param.requires_grad
                
                # param_gbuf_map: param到具体的梯度缓冲区的映射。
                # param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                
                # gbuf_ranges[gbuf_index][dtype][bucket_index]["param_map"][model_param] = 
                # {
                #     "gbuf_world": Range(0, 50),
                #     "gbuf_world_in_bucket": Range(0, 50),
                #     "gbuf_local": Range(0, 50),
                #     "param": Range(0, 50),
                # }
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start : param_range.end
                    ]
                    shard_main_param = shard_model_param.clone().float()  # 复制一份fp32的
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]  # 原先就是fp32的，直接复制引用
                    shard_main_param = shard_model_param
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_main_param)
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

                # bucket和shard optim param_groups的映射关系
                bucket = model_param_to_bucket_map[model_param]
                if bucket not in bucket_to_optim_params_map:
                    # 初始化映射
                    bucket_to_optim_params_map[bucket] = [[] for _ in range(len(opt_group_ranges))]
                # 将shard的参数加入到对应的组中
                bucket_to_optim_params_map[bucket][group_idx].append(shard_main_param)

            # Update optimizer's params.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
            bucket_to_optim_params_map,
        )

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        optim_config: OptimizerConfig,
        model_chunks: Union[DistributedDataParallel, List[DistributedDataParallel]],
        scaler: torch.cuda.amp.GradScaler = None, grad_clip = 0.0,
        data_parallel_group: torch.distributed.ProcessGroup = torch.distributed.group.WORLD,
        data_parallel_group_gloo: torch.distributed.ProcessGroup = None,
        data_parallel_group_idx: int = 0,
    ):
        """
        Distributed optimizer, for all data types (fp16, bf16, and fp32).
        
        分布式优化器，model_chunks表示多个模型（因为可能由虚拟流水线），由DDP包裹
        如果不使用虚拟流水线，那么model_chunks就是一个模型

        The steps in this method create the core mapping between param and grad buffers,
        parameters, and parameter shard ranges, that is needed for converting between model
        param indexes and main parameter shard indexes. This method also updates the optimizer
        parameter groups with the newly created shards.

        Args:
            optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
            config (OptimizerConfig): configuration object for optimizer.
            grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
                this can be None. This case happens when `bf16 = True` and we don't
                use any loss scale. Note that for `bf16 = True`, we can have
                a constant gradient scaler. Also for `bf16 = False`, we
                always require a grad scaler.
            init_state_fn (Callable, optional): function to initialize state in the optimizer.
            per_model_buffers (Dict[int, List[ParamAndGradBuffer]]): the implementation of the
                distributed optimizer is centered on using a contiguous buffer for
                communicating grads & params between the model state and the optimizer state.
                You can find a more detailed description in
                https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md.
            data_parallel_group (torch.distributed.ProcessGroup): data-parallel group to use to
                all-gather params after optimizer.step().
            data_parallel_group_gloo (torch.distributed.ProcessGroup): gloo data-parallel group
                (used in checkpoint loading and saving).
            data_parallel_group_idx (int): index in data-parallel group (used by
                distributed checkpointing logic).
        """

        super().__init__(optimizer, optim_config, scaler, grad_clip)

        # TODO_MY 支持下SGD
        # assert isinstance(
        #     optimizer, Adam
        # ), "Only Adam currently supported, due to checkpointing requirements."
        
        if not isinstance(model_chunks, List):
            model_chunks = [model_chunks]
        
        if self.optim_config.overlap_optim_step:
            # overlap_optim_step需要使用wrapper包裹原hook
            # 基于记录好的cur_comm_hook, cur_comm_hook_args, cur_comm_hook_kwargs来重新构建hook
            # 注意，这里传的是包裹后的optimizer，即self
            for model_chunk in model_chunks:
                model_chunk.register_comm_hook(overlap_optim_step_wrapper(model_chunk.cur_comm_hook, self),
                                               *model_chunk.cur_comm_hook_args, **model_chunk.cur_comm_hook_kwargs)
        
        # Model grad buffer ranges.
        self.per_model_buffers = {}
        for model_idx, model_chunk in enumerate(model_chunks):
            if hasattr(model_chunk, 'buffers'):
                self.per_model_buffers[model_idx] = model_chunk.buffers
        self.buffers = list(itertools.chain(*self.per_model_buffers.values()))
        
        # 构建param到bucket的映射
        self.model_param_to_bucket_map = {}
        for buffer in self.buffers:
            for bucket in buffer.buckets:
                for param in bucket.params:
                    self.model_param_to_bucket_map[param] = bucket
        
        self.data_parallel_group = data_parallel_group
        # 基于dp组，创建新的backend为gloo的dp组
        # 用于后续参数保存的阶段，因为需要在cpu上进行（在gpu上进行的话可能会超显存）
        if data_parallel_group_gloo is not None:
            self.data_parallel_group_gloo = data_parallel_group_gloo
        else:
            world_size = torch.distributed.get_world_size(self.data_parallel_group)
            ranks = list(range(world_size))
            self.group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        # 因为可能有多个dp组，所以需要一个索引来区分
        # 如model_parallel下，2TP-4DP，则有2个DP组，每个DP组中有4个GPU
        self.data_parallel_group_idx = data_parallel_group_idx
        
        self.gbuf_idx_to_model_idx_map = {}
        gbuf_idx = 0
        for model_idx, buffers in self.per_model_buffers.items():
            for _ in buffers:
                self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
                gbuf_idx += 1
                
        # 构建每个dp组中，每个rank中参数所对应的buffer的区域
        self.gbuf_ranges = []
        self.per_bucket_numel = []
        self.per_bucket_numel_unpadded = []
        for buffer in self.buffers:

            self.per_bucket_numel.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.grad_data.numel() for bucket in buffer.buckets
                    ]
                }
            )
            self.per_bucket_numel_unpadded.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.numel_unpadded for bucket in buffer.buckets
                    ]
                }
            )
            self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))
        # param到具体的梯度缓冲区的映射。
        # 基于model_param_gbuf_map和gbuf_ranges，则可以根据一个param，找到其对应的buffer区域
        # self.model_param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

        # Optimizer ranges.
        # self.model_param_group_index_map, 某个dp rank，参数 -> 所处的组id和组内顺序
        # self.opt_group_ranges, 某个dp rank所包含的参数组，其中的参数还没有进行shard
        (
            self.model_param_group_index_map,
            self.opt_group_ranges,
        ) = self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

        # Allocate main param shards.
        # 会将opt_group_ranges中的f16参数替换为f32参数，覆盖到opt_group_ranges[group_id]["orig_group"]["params"]
        # 现在opt_group_ranges中的group包含两种参数：
        # 1. 原模型参数，没进行shard，在opt_group_ranges[group_id]["param"]中
        # 2. 转换为f32的参数，进行了shard，在opt_group_ranges[group_id]["orig_group"]["params"]中
        #* 同时构建bucket和shard optim param_groups的映射关系，用于overlap_optim_step
        #* self.bucket_to_optim_params_map = {bucket: [group_params, ...]}
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
            self.bucket_to_optim_params_map,
        ) = self._build_model_and_main_param_groups(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, self.model_param_to_bucket_map
        )

        # Now construct data structures to manage all-gather handles.
        # 构建索引，来处理all-gather handles
        self.all_gather_handles = []
        self.all_gather_handle_index_to_bucket_index_map = []  # 一对一，一个handle对应一个bucket
        self.model_index_to_all_gather_handle_index_map = {}  # 一对多，一个model chunk有多个bucket
        self.all_gather_handle_indices = []  # 好像没用到，因为indices就是0, 1, 2, ...的顺序
        self.param_to_all_gather_handle_index_map = {}  # 多对一，多个param对应一个handle，因为一个bucket有多个param

        # 这里pbuf_view_items的顺序是，forward顺序，即buffer中的bucket以forward顺序排列
        # 注意，原buffer中的bucket是backwrard顺序排列的，这里因为是all_gather，所以需要forward顺序
        # self.pbuf_view_items: [(gbuf_index, dtype, bucket_index, bucket.param_data, buf_views), ...]
        self.pbuf_view_items = self._get_model_param_buffer_dp_views()
        for (gbuf_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
            self.all_gather_handle_index_to_bucket_index_map.append(
                (gbuf_index, dtype, bucket_index)
            )
            all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
            self.all_gather_handles.append(None)

            # Store all all_gather_handle_indices.
            model_idx = self.gbuf_idx_to_model_idx_map[gbuf_index]
            if model_idx not in self.model_index_to_all_gather_handle_index_map:
                self.model_index_to_all_gather_handle_index_map[model_idx] = []
            self.model_index_to_all_gather_handle_index_map[model_idx].append(
                all_gather_handle_index
            )

            for param in self.buffers[gbuf_index].buckets[bucket_index].params_list:
                self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
        self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

        # 在model0进行forward时，对model1的参数进行gather，计算与通信重叠
        self.overlap_param_gather = self.optim_config.overlap_param_gather
        self.remove_pre_hook_handle = None  # 处理register_module_forward_pre_hook的返回值，保存下来，用于remove
        if self.overlap_param_gather:
            self.enable_pre_hook()

        # optim参数更新成功后，才会进行all-gather
        self.update_successful = False

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        # 将shard好后的group赋值给optimizer的param_groups中
        # 现在optimizer.param_groups中的是转换为f32的参数，进行了shard，在opt_group_ranges[group_id]["orig_group"]["params"]中
        # opt_group_ranges[group_id]["orig_group"]也包含了所有原optim的超参数，只有params不一样，是经过shard的
        self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        # 因为optimizer.param_groups中的参数是shard的，所以需要将state中的参数也转换为shard，通过以下方法实现
        # load_state_dict只会载入param_groups中对应的状态
        # TODO_MY 在optim的state非空时好像有问题，因为torch的otim的state_dcit()要求state和param_groups对应的参数一致
        # TODO_MY 但是这里的param_groups中的参数是shard的，而state中的参数是原始的，所以会有问题
        # self.optimizer.load_state_dict(self.optimizer.state_dict())  # 一开始，optim上param对应的优化器状态还是空的，所以应该不需要这一步
        
        
        
        # 收集所有需要unscale梯度的参数，用于grad_clip
        self._post_init()

    def enable_pre_hook(self):
        """
        Enable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        assert self.remove_pre_hook_handle is None
        # 全局hook，对模型中的所有module都起作用
        # 假设有两个顺序module，module1, module2
        # 在module1进行forward前，会调用hook，对module2中的参数进行all-gather，计算与通信重叠
        #! 注意，因为用的module.parameters(recurse=False)来遍历
        #!   所以model中最好不要使用nn.Parameter，否则可能会打断all-gather的流程
        self.remove_pre_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(
            self._make_forward_pre_hook()
        )

    def disable_pre_hook(self):
        """
        Disable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        assert self.remove_pre_hook_handle is not None
        self.remove_pre_hook_handle.remove()
        self.remove_pre_hook_handle = None

        # Make sure all-gathers are completed as needed.
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=True)

    def _get_model_param_range_map(self, param: torch.nn.Parameter):
        """
        Given a model param, get the index sub-range of the param that this
        data-parallel rank owns.
        """
        gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[param]
        gbuf_range_map = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """
        With the distributed optimizer, the model parallel group is the
        entire world.
        """
        return None

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """

        state_dict = {}

        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {
            k: v for k, v in self.optimizer.state_dict().items() if k != "state"
        }
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer odering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [
            {**group, "params": list(inner_state_dict["param_groups"][idx]["params"]),}
            for idx, group in enumerate(state_dict["optimizer"]["param_groups"])
        ]

        # Allocate 'dummy' data for optimizer state (i.e., torch.empty() below)
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for gbuf_range_maps in self.gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Get parameter ordering information (see method docstring
                        # for details).
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        state_order = inner_state_dict["param_groups"][group_index]["params"][
                            group_order
                        ]

                        # Allocate dummy tensors.
                        numel = len(param_range_map["gbuf_world"])
                        init_shard = lambda: torch.empty(
                            (numel,), dtype=torch.float32, device=torch.cuda.current_device()
                        )

                        state_dict_state.append(
                            (state_order, {"exp_avg": init_shard(), "exp_avg_sq": init_shard(),})
                        )

        # Sort by state order (see method docstring for details).
        state_dict_state.sort(key=lambda s: s[0])
        state_dict_state = {s[0]: s[1] for s in state_dict_state}

        # Optimizer.
        self.optimizer.load_state_dict(
            {"state": state_dict_state, "param_groups": state_dict_param_groups,}
        )

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        if 'param_state' in state_dict:
            assert 'param_state_sharding_type' in state_dict, state_dict.keys()
            param_state = state_dict['param_state']
            sharding_type = state_dict['param_state_sharding_type']
            logger.info(f'Loading distributed optimizer sharded state of type {sharding_type}')
            if sharding_type == 'dp_zero_gather_scatter':
                self.load_parameter_state_from_dp_zero(param_state)
            elif sharding_type == 'fully_sharded_bucket_space':
                self.load_parameter_state_from_fs_bucket_space(param_state)
            else:
                raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

    def get_parameter_state_fs_bucket_space(self):
        """Get internal representation of parameter state without any copies and modifications.

        This is referred to as "fully sharded bucket space" because the optimizer state is
        fully sharded (e.g. no gather involved) and bucket-centric (the state
        follows the internal structure of the Distributed Optimizer buckets)
        as opposed to model-centric (typical structure of PyT optimizers)
        """
        state = {
            "per_bucket_numel": self.per_bucket_numel,
            "per_bucket_numel_unpadded": self.per_bucket_numel_unpadded,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buckets_state = []
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = []
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                            "gbuf_local_start": param_range_map["gbuf_local"].start,
                            "gbuf_local_end": param_range_map["gbuf_local"].end,
                        }
                        bucket_state.append(tensors)
                    buckets_state.append(bucket_state)
                dtype_state[dtype] = buckets_state
            state[gbuf_idx] = dtype_state
        return state

    def get_parameter_state_dp_zero(self):
        """Get parameter state (i.e., parameter & optimizer tensors).

        This method performs two steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Collect param states.
        state = {
            "buckets_coalesced": True,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                # Create coalesced tensors for all state related to parameters in this buffer.
                # 只保存没有pad的部分
                world_tensors = {}
                if data_parallel_rank == 0:
                    world_tensors = {
                        key: torch.empty(
                            (buffer_numel_unpadded,), dtype=torch.float32, device="cpu"
                        )
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }
                    world_tensors["numel_unpadded"] = buffer_numel_unpadded
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    # 对每个bucket操作，获取rank各自的local_shards
                    # 然后聚合到rank0上进行保存，是在cpui进行聚合的，需要使用gloo后端

                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    gbuf_world_numel_unpadded = (
                        self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                    )
                    assert gbuf_world_numel_unpadded <= gbuf_world_numel

                    local_shards = {
                        key: torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                        }

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                                tensors[key].detach().cpu()
                            )

                    # Gather contiguous shards on DP rank 0.
                    for key, send_tensor in local_shards.items():

                        # Gather tensor list.
                        if data_parallel_rank == 0:
                            recv_tensors = [
                                torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        # Gather.
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Concatenate.
                        if data_parallel_rank == 0:
                            recv_tensors_concatenated = torch.cat(recv_tensors)
                            # Copy this bucket's collected all-gather tensors into the right place in the
                            # tensor for the buffer. The tensor for the buffer gets rid of the padding
                            # between buckets.
                            # padding的部分没必要复制
                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            world_tensors[key][start:end].copy_(
                                recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                            )

                    offset_in_world_tensors += gbuf_world_numel_unpadded

                # Collect world state.
                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter state on DP rank 0.

        Args:
            filename (str): path to save parameter state to.
        """

        state_dict = self.get_parameter_state_dp_zero()
        if torch.distributed.get_rank(self.data_parallel_group) == 0:  # 每个dp组的第一个local rank
            torch.save(state_dict, filename)

    def load_parameter_state_from_fs_bucket_space(self, state_dict):
        """ Loads the parameter state from an internal representation.

        Inverse of the `get_parameter_state_internal_repr` method.
        """
        if state_dict is not None and "per_bucket_numel_unpadded" in state_dict:
            per_bucket_numel_unpadded_in_checkpoint = state_dict["per_bucket_numel_unpadded"]
            assert self.per_bucket_numel_unpadded == per_bucket_numel_unpadded_in_checkpoint, (
                f"Number of unpadded elements in each bucket need to be the same in current run "
                f"({self.per_bucket_numel_unpadded}) and checkpoint "
                f"({per_bucket_numel_unpadded_in_checkpoint})"
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = state_dict[gbuf_idx][dtype][bucket_idx]

                    # State dict bucket state can be 1 entry longer in case of padding
                    assert len(bucket_state) in (
                        len(gbuf_range_map["param_map"]),
                        len(gbuf_range_map["param_map"]) + 1,
                    ), (len(bucket_state), len(gbuf_range_map["param_map"]))
                    for src_tensors, (model_param, param_range_map) in zip(
                        bucket_state, gbuf_range_map["param_map"].items()
                    ):
                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        dst_tensors = {
                            "param": main_param,
                            **optim_state,
                        }
                        for key in dst_tensors:
                            dst_tensors[key].copy_(src_tensors[key])

    def load_parameter_state_from_dp_zero(self, state_dict):
        """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank,
        using the new checkpoint format with coalesced state across buckets.

        This method performs the reverse of get_parameter_state_dp_zero():
        - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
          rank receives its relevant subset of the world buffers).
        - For each DP rank, copy param & optimizer shards from contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Scatter tensors to all DP ranks.
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                if data_parallel_rank == 0:
                    buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                    checkpoint_numel_unpadded = state_dict[gbuf_idx][dtype]["numel_unpadded"]
                    assert buffer_numel_unpadded == checkpoint_numel_unpadded, (
                        f"Number of unpadded elements must be same in current run "
                        f"({buffer_numel_unpadded}) and checkpoint ({checkpoint_numel_unpadded})"
                    )
                for key in ("param", "exp_avg", "exp_avg_sq"):
                    offset_in_world_tensors = 0
                    for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                        # Compute local DP contiguous shard's size.
                        gbuf_world_numel = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                        )
                        assert gbuf_world_numel % data_parallel_world_size == 0
                        gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                        gbuf_world_numel_unpadded = (
                            self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                        )
                        assert gbuf_world_numel_unpadded <= gbuf_world_numel

                        # Contiguous local shards (received from DP rank 0).
                        recv_tensor = torch.empty(
                            (gbuf_local_numel,), dtype=torch.float32, device="cpu"
                        )

                        # Scatter tensor list.
                        if data_parallel_rank == 0:
                            world_tensors = state_dict[gbuf_idx][dtype][key]

                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            assert 0 <= start < end <= world_tensors.numel()
                            world_tensor = world_tensors[start:end]
                            offset_in_world_tensors += gbuf_world_numel_unpadded

                            # Pad world_tensor to gbuf_world_numel. Don't pad at the front, pad at the back.
                            world_tensor = torch.nn.functional.pad(
                                world_tensor, (0, gbuf_world_numel - gbuf_world_numel_unpadded)
                            )
                            assert world_tensor.numel() == gbuf_world_numel
                            gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                            send_tensors = [
                                world_tensor[i : (i + gbuf_local_numel)] for i in gbuf_start_idxs
                            ]
                        else:
                            send_tensors = None

                        # Scatter.
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Copy local contiguous shards to param/optim shards.
                        for model_param, param_range_map in gbuf_range_map["param_map"].items():

                            # Main param & optimizer states.
                            group_index, group_order = self.model_param_group_index_map[model_param]
                            main_param = self.optimizer.param_groups[group_index]["params"][
                                group_order
                            ]
                            if key == "param":
                                tensor_to_copy_into = main_param
                            else:
                                optim_state = self.optimizer.state[main_param]
                                tensor_to_copy_into = optim_state[key]

                            # Copy states into contiguous shard.
                            gbuf_local_start = param_range_map["gbuf_local"].start
                            gbuf_local_end = param_range_map["gbuf_local"].end
                            tensor_to_copy_into.data.copy_(
                                recv_tensor[gbuf_local_start:gbuf_local_end]
                            )

    def load_parameter_state(self, filename: str):
        """Load the distributed parameter state from disk.

        Args:
            filename (str): path to load parameter state from.
        """
        state_dict = None
        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            state_dict = torch.load(filename)

        self.load_parameter_state_from_dp_zero(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.
        
        这里会发起第一次的all-gather，同时只会将dp rank相对应的grads和optim中的grads置零

        Args:
            set_to_none (bool): if true, set grads to None.
        """
        # TODO_MY 这里应该只需要对self.shard_fp32_from_float16_groups的grad进行zero?
        # 因为其他四个的grad都是model的grad，并且前面已经对model grad进行了zero
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_fp32_from_float16_groups,
        ):
            for group in groups:
                zero_grad_group_helper(group, set_to_none)

        # If overlapping param all-gather with forward compute, launch all-gather
        # for first accessed bucket here before forward compute is initiated.
        # The all-gather for the next bucket will be launched in the forward
        # pre-hook when this all-gather finishes (to ensure that the communication
        # kernels don't head-of-line block the compute kernels since we run with
        # CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence parallelism).
        # 这里会发起第一次的all-gather，后续的all-gather会在forward pre-hook中发起
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)

    def _get_model_param_buffer_dp_views(self):
        """
        Get shard views of each of the param buffers.

        In this nested list, the top level is grouped by the virtual model
        index and the buffer's data type. The sub-level is a list of
        shards of that buffer, where each shard in the list represents
        a contiguous view of the buffer, that is owned by a data-parallel
        rank. The shard boundary does not respect parameter boundaries, and
        so the elements of some parameters are split across data parallel
        ranks.

        Additionally, return references to the entire buffers, for use
        in _all_gather_base.
        """

        # Buffer views.
        # Add in reverse order in each model chunk since buckets start from the end of the model but we want
        # all-gathers to run first for the start of the model (same order as forward pass).
        # buffer中的bucket是倒序的（backward顺序），但是all-gathers需要正序，所以再进行一个倒序，view_items_per_model_chunk中的bucket是forward顺序
        
        # We keep the view_items in model chunk order since we want to still first run all_gather and
        # all_gather_handle.wait() for the first model chunk.
        # model chunk的顺序还是正序，因为在虚拟流水线中，先进行virtual1阶段再virtual2阶段
        
        # In all cases, we want all_gather and all_gather_handle.wait() to be called in the same order,
        # and all_gather_handle.wait() needs to be called just before the corresponding forward pass.
        # all_gather_handle.wait()需要在forward之前调用
        view_items = []
        for gbuf_index, buffer in enumerate(self.buffers):  # self.buffers是所有buffer的列表，包括f32和f16和其他model chunk的
            view_items_per_model_chunk = []
            dtype = self.buffers[gbuf_index].param_dtype
            for bucket_index, bucket in enumerate(buffer.buckets):
                data_parallel_world_size = torch.distributed.get_world_size(
                    self.data_parallel_group
                )
                buf_views = shards_buffer(bucket.param_data, data_parallel_world_size)
                view_items_per_model_chunk.insert(
                    0, (gbuf_index, dtype, bucket_index, bucket.param_data, buf_views)
                )
            view_items.extend(view_items_per_model_chunk)

        return view_items

    def _dispatch_gather_model_params(self, all_gather_handle_index: int, force_sync: bool = False):
        """
        All-gather updated model params.

        When using the distributed optimizer, the params are already laid out in a contiguous
        buffer (see mcore/distributed/param_and_grad_buffer.py for details), and so the
        all-gather will put the results in the right region of memory.
        """
        async_op = self.overlap_param_gather and not force_sync
        # 在update_successful=True的情况下，才会进行all-gather
        if self.update_successful:
            data_parallel_group = self.data_parallel_group
            data_parallel_rank = torch.distributed.get_rank(data_parallel_group)  # local rank

            # All-gather updated main params.
            # All param_buf views are guaranteed to have the same number of elements
            # across all data-parallel ranks, due to padding done in
            # param_and_grad_buffer.py). Thus, all sub-views will have consistent
            # start / end indexes across data-parallel ranks.
            # 这个all-gather对应的bucket
            (gbuf_index, dtype, bucket_index, pbuf, pbuf_views) = self.pbuf_view_items[
                all_gather_handle_index
            ]
            assert all_gather_handle_index < len(self.all_gather_handles)
            # all-gather到DDP中创建好的连续buffer中
            all_gather_handle = torch.distributed._all_gather_base(
                pbuf, pbuf_views[data_parallel_rank], group=data_parallel_group, async_op=async_op,
            )
            self.all_gather_handles[all_gather_handle_index] = all_gather_handle
            assert self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index] == (
                gbuf_index,
                dtype,
                bucket_index,
            )

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather)
        and then copy the results from the param_buffer into model_params.
        """

        def hook(module, *unused):
            assert (
                self.overlap_param_gather
            ), "Should use pre-hook only when overlap_param_gather is True"

            # Make sure all parameters in this module have been all-gathered as necessary.
            # 一个module中的参数可能会分在不同的bucket中，所以要确保所有参数的bucket都完成了all-gather
            # 每个bucket完成后，都会立刻对后续的bucket进行all-gather，同时这个module开始forward
            # 从而实现计算与通信的重叠
            #! 注意，因为用的module.parameters(recurse=False)来遍历
            #!   所以model中最好不要使用nn.Parameter，否则可能会打断all-gather的流程
            for param in module.parameters(recurse=False):
                # Skip parameters that don't require grad.
                if not param.requires_grad:
                    continue

                # Some params might be handled in another DistributedOptimizer instance; for
                # example, we use separate DistributedOptimizer instances for expert and
                # non-expert params.
                if param in self.param_to_all_gather_handle_index_map:
                    all_gather_handle_index = self.param_to_all_gather_handle_index_map[param]
                    self._finish_param_sync_helper(all_gather_handle_index)

        return hook

    def finish_param_sync(self, model_index: int, *unused):
        """
        Finishes all necessary param syncs for the model_index'th model chunk.

        Args:
            model_index (int): index of model chunk to synchronize params.
        """
        if model_index not in self.model_index_to_all_gather_handle_index_map:
            return

        all_gather_handle_indices = self.model_index_to_all_gather_handle_index_map[model_index]
        for all_gather_handle_index in all_gather_handle_indices:
            self._finish_param_sync_helper(all_gather_handle_index)

    def _finish_param_sync_helper(self, all_gather_handle_index: int):
        """
        Waits on all_gather_handle if necessary, then dispatches the next all-gather
        as necessary.
        """

        # First check if there is an outstanding all-gather handle for this param.
        # If so, wait on the handle to ensure the communication is finished.
        assert all_gather_handle_index < len(self.all_gather_handles)
        all_gather_handle = self.all_gather_handles[all_gather_handle_index]
        if all_gather_handle is not None:
            all_gather_handle.wait()  # 异步等待当前参数的all-gather完成，因为当前参数的all-gather是由前面的module触发的
            self.all_gather_handles[all_gather_handle_index] = None

            # Launch the all-gather for the next bucket now.
            # We can't pre-launch all-gathers for all buckets at once since we don't
            # want to head-of-line block the compute kernels with communication kernels
            # (since we run with CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence
            # parallelism).
            # 发起下一个bucket的all-gather
            next_all_gather_handle_index = all_gather_handle_index + 1
            if next_all_gather_handle_index < self.num_all_gather_handles:
                self._dispatch_gather_model_params(next_all_gather_handle_index)

    def _collect_main_params_for_unscaling_grads(self):
        """
        收集需要unscale梯度的参数，即需要更新的参数，后续将其中的grads进行unscale
        """
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but written differently, so the two should be combined.
        """
        return [
            param.grad.data for group in self.optimizer.param_groups for param in group["params"]
        ]

    def _get_model_and_main_params_data_float16(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.shard_float16_groups, self.shard_fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)  # 不同param
        copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)  # 同一个param，将同一个参数的main_grad复制到shard_grad中

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        # copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)  # 空间共享，不需要复制

    # TODO_MY 微调用
    # def _copy_model_params_to_main_params(self):
    #     """
    #     Copy model params to main params.

    #     During finetuning, this method is used to reload the main params from
    #     the model params. This copy does not make use of the grad buffer as
    #     an intermediary.
    #     """

    #     # Utility method for copying group params.
    #     def copy_group_params(model_groups, shard_main_groups):
    #         for model_group, shard_main_group in zip(model_groups, shard_main_groups):
    #             for model_param, shard_main_param in zip(model_group, shard_main_group):

    #                 param_range_map = self._get_model_param_range_map(model_param)
    #                 param_range = param_range_map["param"]
    #                 assert param_range.size == shard_main_param.nelement()

    #                 shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
    #                 shard_main_param.data.copy_(shard_model_param)

    #     # Copy model groups to shard groups.
    #     copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
    #     copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

    def _reset_metadata_and_sync_gather_all_model_params(self, force_sync: bool):
        """
        Reset metadata needed to track results of all-gathers.
        """
        self.all_gather_handles = [None for _ in range(len(self.all_gather_handles))]

        # Launch synchronous all-gather if --overlap-param-gather is turned on or if force_sync
        # is explicitly set to True (e.g., if we are going to turn off all-gather overlapping for
        # validation / test iterations).
        # --overlap-param-gather下不会立即发起all-gather，而是在forward pre-hook中发起
        if not self.overlap_param_gather or force_sync:
            for all_gather_handle_index in range(self.num_all_gather_handles):
                self._dispatch_gather_model_params(all_gather_handle_index, force_sync=force_sync)

    @torch.no_grad()
    def step(self):
        """
        Step optimizer.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        self.update_successful = super().step()  # 更新成功才会进行后续的all-gather

        # If not overlapping all-gather for parameters, launch synchronous all-gather
        # communication calls here. If overlapping all-gather for parameters, the following
        # call to _gather_all_model_params is a no-op: the first all-gather is launched
        # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
        # are launched in the forward pre-hook.
        # 若没用overlap，则在这里发起同步的all-gather（在update_successful=True的情况下）
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
