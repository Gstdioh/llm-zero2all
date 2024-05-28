import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, bf16_compress_wrapper
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
import torch.distributed as dist
import torch.profiler


# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

# NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 test.py

process_group = dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
master_process = (ddp_rank == -1 or ddp_rank == 0)
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)

seq_len = 1024
input_dim = 1024 * 4
hidden_dim = 1024 * 4
output_dim = 1024

static_input = torch.randn(seq_len, input_dim).to(device)
static_target = torch.randn(seq_len, output_dim).to(device)

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
).to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=0.01)

# Warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    model = DDP(model, device_ids=[ddp_local_rank])
    powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
                        warm_start=True, use_error_feedback=True, start_powerSGD_iter=5, 
                        min_compression_rate=0.5, orthogonalization_epsilon=1e-6)
    model.register_comm_hook(powerSGD_state, powerSGD_hook)
    # model.register_comm_hook(powerSGD_state, bf16_compress_wrapper(powerSGD_hook))
    # model.register_comm_hook(process_group, test_hook)

    for i in range(20):
        optim.zero_grad(set_to_none=True)
        output = model(static_input)
        loss = loss_fn(output, static_target)
        loss.backward()
        torch.cuda.synchronize()
        optim.step()
torch.cuda.current_stream().wait_stream(s)

if master_process:
    print("capturing")

with torch.profiler.profile():
    pass

# capture
g = torch.cuda.CUDAGraph()
optim.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_output = model(static_input)
    static_loss = loss_fn(static_output, static_target)
    static_loss.backward()

real_inputs = torch.randn(seq_len, input_dim).to(device)
real_targets = torch.randn(seq_len, output_dim).to(device)

torch.cuda.synchronize()

if master_process:
    print("starting")

# profile
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=0,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('../../res_profile/test_graph/10_Power_SGD_no_block_graph', worker_name=f'rank{ddp_rank}'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    
    for i in range(10):
        static_input.copy_(real_inputs)
        static_target.copy_(real_targets)
        
        # You don't even need to call optimizer.zero_grad() between iterations
        # because the captured backward refills static .grad tensors in place.
        g.replay()
        torch.cuda.synchronize()
        optim.step()
        
        if master_process:
            print(i)
        
        torch.cuda.synchronize()
        
        p.step()
