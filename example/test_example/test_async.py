import os

import torch
import torch.distributed as dist
import torch.profiler


os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

def test_com(tensor):
    com_handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM).get_future()

    com_handle.then(lambda x: print(x))

# NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 mytest.py

dist.init_process_group(backend="nccl")

ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)

# communication
tensor_com1 = torch.randn(1024, 1024).to(device)
tensor_com2 = torch.randn(1024, 1024).to(device)

# computation
tensor1 = torch.randn(1024*4, 1024*4).to(device)
tensor2 = torch.randn(1024*4, 1024*4).to(device)

async_comm = False

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
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./res_profile/test_comm/big_compute_1', worker_name=f'rank{ddp_rank}'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    
    for i in range(10):
        #  sync_time: 3.948ms
        # async_time: 3.617
        if i == 5:
            async_comm = True
        
        # communication
        # 调用kernel：ncclKernel_AllReduce_RING_LL_Sum_float(ncclWorkElem)
        dist.all_reduce(tensor_com1, op=dist.ReduceOp.SUM, async_op=async_comm)  # kernel耗时：3.443ms
        
        # computation
        # 包含四个操作：
        # 1. cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
        # 2. cudaEventQuery
        # 3. cudaMemsetAsync
        # 4. cudaLaunchKernel
        res1 = tensor1 @ tensor2  # kernel耗时：0.128ms
        
        dist.all_reduce(tensor_com2, op=dist.ReduceOp.SUM, async_op=async_comm)
        
        res1 = tensor1 @ tensor2  # kernel耗时：0.128ms
        
        torch.cuda.synchronize()
        
        print(i)  # 耗时：0.02ms
        
        p.step()
