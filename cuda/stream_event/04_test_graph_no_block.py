import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, bf16_compress_wrapper
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
import torch.distributed as dist
import torch.profiler


# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 test_graph.py

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

input_data = torch.randn(seq_len, input_dim).to(device)
label_data = torch.randn(seq_len, output_dim).to(device)

model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
).to(device)
loss_fn = nn.MSELoss()
optim = torch.optim.AdamW(model.parameters(), lr=0.01)
model = DDP(model, device_ids=[ddp_local_rank])

def test_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        input_tensor = bucket.buffer()
        
        all_input_tensors = input_tensor.clone()

        fut = dist.all_reduce(
            all_input_tensors, group=process_group, async_op=True
        ).get_future().wait()[0]
        
        tensor = input_tensor * fut

    fut = torch.futures.Future()
    fut.set_result(torch.zeros_like(tensor, device=tensor.device))

    return fut

# powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
#                     warm_start=True, use_error_feedback=True, start_powerSGD_iter=5, 
#                     min_compression_rate=0.5, orthogonalization_epsilon=1e-6)
# model.register_comm_hook(powerSGD_state, powerSGD_hook)
# model.register_comm_hook(powerSGD_state, bf16_compress_wrapper(powerSGD_hook))
model.register_comm_hook(process_group, test_hook)

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
    on_trace_ready=torch.profiler.tensorboard_trace_handler('../../res_profile/test_graph/04_grad_hook_no_block', worker_name=f'rank{ddp_rank}'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    
    for i in range(10):
        optim.zero_grad(set_to_none=True)
        output = model(input_data)
        loss = loss_fn(output, label_data)
        loss.backward()
        
        torch.cuda.synchronize()
        
        optim.step()
        
        if master_process:
            print(i)
        
        p.step()
