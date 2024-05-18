import torch
import torch.profiler
import torch.nn as nn


model = nn.Linear(5, 2).cuda()

x = torch.randn(1, 5).cuda()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=8),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./res_profile_test/test', worker_name='worker0'),
    record_shapes=True,
    profile_memory=True,  # This will take 1 to 2 minutes. Setting it to False could greatly speedup.
    with_stack=True
) as p:
    
    for i in range(100):
        print(i)
        
        y = model(x)
    
        p.step()
        
        if i > 12:
            break