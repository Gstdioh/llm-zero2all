# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel."""

    grad_reduce_in_fp32: bool = False
    """If true, reduce grads in fp32."""

    overlap_grad_reduce: bool = False
    """
    If true, overlap grad all-reduce / reduce-scatter with backward compute.
    
    同步：设置bucket_size=None，即所有参数在同一个bucket中，计算完成后进行通信
    重叠：设置相应的bucket_size大小，分为多个bucket，每进行完一部分param的计算就进行异步通信，计算和通信重叠
    """

    use_distributed_optimizer: bool = False
    """If true, issue reduce-scatter collectives to aggregate gradients and clean up originally
       allocated model parameters, otherwise issue all-reduce collectives.
    """

    check_for_nan_in_grad: bool = False
    """ If true, check for NaNs in gradients _before_ communication collective."""

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
    value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger buckets
    to ensure collectives do not become latency-bound)."""
