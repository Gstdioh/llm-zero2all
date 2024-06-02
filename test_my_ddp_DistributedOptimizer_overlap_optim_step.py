import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, bf16_compress_wrapper
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
import torch.distributed as dist
import torch.profiler

from parallel.distributed_data_parallel import DistributedDataParallelConfig
from parallel.distributed_data_parallel import DistributedDataParallel as MyDDP
from optimizer import OptimizerConfig, FP32Optimizer, Float16OptimizerWithFloat16Params
from parallel.distributed_optimizer import DistributedOptimizer


# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 test_my_ddp_DistributedOptimizer_overlap_optim_step.py

process_group = dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
master_process = (ddp_rank == -1 or ddp_rank == 0)
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)

device_type = "cuda"
dtype = "bfloat16"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# 设置自动混合精度的上下文
ctx = (
    nullcontext()
    if device_type == "cpu"
    # else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # 原来的代码
    else torch.autocast(device_type=device_type, dtype=ptdtype)
)

torch.manual_seed(42)

seq_len = 1024
input_dim = 1024 * 4
hidden_dim = 1024 * 4
output_dim = 1024 * 4

input_data = torch.randn(seq_len, input_dim).to(device).to(ptdtype)
label_data = input_data * 2 + 1

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
).to(device)
model = model.to(ptdtype)
loss_fn = nn.MSELoss()

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
grad_clip = 0.0
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# -------------------------------------------------------------------
# 测试我的DDP
# model = DDP(model, device_ids=[ddp_local_rank])
ddp_config = DistributedDataParallelConfig(
    grad_reduce_in_fp32=False,
    overlap_grad_reduce=True,
    use_distributed_optimizer=True,
    check_for_nan_in_grad=False,
    bucket_size=4_000_000)
model = MyDDP(model,
              ddp_config,
              data_parallel_group=process_group,
              disable_bucketing=False)

# 构建分布式优化器
optim_config = OptimizerConfig(
    precision_dtype=dtype,
    overlap_optim_step=True,
    overlap_zero_grad_buffer=True,
    use_distributed_optimizer=True,
    overlap_param_gather=True)
optimizer = DistributedOptimizer(optimizer, optim_config, model, scaler=scaler, grad_clip=grad_clip)

# profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=0,
        warmup=2,
        active=2,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./res_profile/test_my_ddp/03_DistributedOptimizer_overlap_optim_step_grad_bfloat16_fix_3',
                                                            worker_name=f'rank{ddp_rank}'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
        
    for i in range(6):
        with ctx:
            output = model(input_data)
            loss = loss_fn(output, label_data)
        scaler.scale(loss).backward()
        
        torch.cuda.synchronize()
        
        optimizer.step()  # 内部自动根据overlap_optim_step来选择是否要更新参数
        
        optimizer.zero_grad(set_to_none=True)  # 内部自动根据overlap_optim_step来选择是否要清空梯度

        torch.cuda.synchronize()
        
        if master_process:
            print(f"iter {i}, loss: {loss:.4f}, scale: {scaler.get_scale()}")
        
        p.step()
