from typing import Dict
import os

import torch
import torch.distributed


def copy_tensor_to_device_in_object(obj, device):
    '''
    将对象中的所有张量复制到指定的设备上
    
    需要给PowerSGDState添加copy_tensor_to_device方法
    '''
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = copy_tensor_to_device_in_object(value, device)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = copy_tensor_to_device_in_object(value, device)
    elif hasattr(obj, "copy_tensor_to_device"):
        obj.copy_tensor_to_device(device)
    return obj


def save_checkpoint(exp_global: Dict, prefix="best"):
    """
    保存状态，该函数会保存两个，例如一个为最优，一个为次优；或者一个为最新，一个为次新
    防止保存时突然中断，导致文件损坏，所以这里保存两个副本
    """
    # 初始化参数
    use_powerSGD_hook = exp_global["use_powerSGD_hook"]
    ddp_local_rank = exp_global["ddp_local_rank"]
    out_dir = exp_global["out_dir"]
    ddp = exp_global["ddp"]
    master_process = exp_global["master_process"]
    raw_model = exp_global["raw_model"]
    optimizer = exp_global["optimizer"]
    iter_num = exp_global["iter_num"]
    best_val_loss = exp_global["best_val_loss"]
    grad_buffer_is_powerSGD_error = exp_global["grad_buffer_is_powerSGD_error"]
    model = exp_global["model"]
    ddp_rank = exp_global["ddp_rank"]
    
    if ddp and use_powerSGD_hook:
        powerSGD_state = exp_global["powerSGD_state"]
    if ddp_local_rank == 0:
        reslog = exp_global["reslog"]
    
    # 保存PowerSGD的状态前，需要将process_group设置为None，因为process_group不能被序列化
    if ddp and use_powerSGD_hook:
        # 保存process_group有问题：https://discuss.pytorch.org/t/how-to-resume-with-powersgd-enabled-training/148747/2
        pg = powerSGD_state.process_group
        powerSGD_state.process_group = None
    
    # 为了防止保存最优时程序中断，即最优保存失败，需要保存一个次优版本
    # 先将次优(best2)删除，然后将最优(best1)改名为次优
    best1_prefix = f"{prefix}1_"
    best2_prefix = f"{prefix}2_"
    
    ckpt_out_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_out_dir, exist_ok=True)
    
    # 只有每个节点的local rank0需要进行删除和重名操作
    if ddp_local_rank == 0:
        # 先将ckpt_out_dir下的次优前缀文件删除
        for file_basename in os.listdir(ckpt_out_dir):
            if file_basename.startswith(best2_prefix):
                os.remove(os.path.join(ckpt_out_dir, file_basename))
        # 然后将最优(best1)改名为次优
        for file_basename in os.listdir(ckpt_out_dir):
            if file_basename.startswith(best1_prefix):
                new_file_basename = best2_prefix + file_basename[len(best1_prefix):]
                os.rename(os.path.join(ckpt_out_dir, file_basename), os.path.join(ckpt_out_dir, new_file_basename))
    
    if ddp:
        # 等待所有rank都删除和重名完
        torch.distributed.barrier()
    
    # 保存完次优后，可以放心进行保存最优了
    # rank0需要保存的文件
    if master_process:
        # 1. 单独保存模型权重文件
        torch.save(raw_model.state_dict(), os.path.join(ckpt_out_dir, best1_prefix + "model.pt"))
        # 2. 保存训练状态
        checkpoint = {
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            # "powerSGD_state": powerSGD_state,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),  # 获取当前的RNG状态，使得resume后的随机数和当前一致（即考虑到dropout等操作的影响）
        }
        torch.save(checkpoint, os.path.join(ckpt_out_dir, best1_prefix + "ckpt.pt"))
        checkpoint = None  # 保存完，可以删除引用了
        
    # 所有rank都需要保存的文件，powerSGD_state
    if ddp and use_powerSGD_hook:
        powerSGD_state_dict = {}
        powerSGD_state_dict["powerSGD_state"] = powerSGD_state
        # 如果powerSGD_grad_buffer_is_error=True，则error_dict状态在grad_buffer中
        if grad_buffer_is_powerSGD_error:
            powerSGD_state_dict["powerSGD_state"].error_dict = {}  # error_dict保存在grad_buffer中
            grad_buffers = []
            for buffer in model.buffers:
                grad_buffers.append(buffer.grad_data)
            powerSGD_state_dict["grad_buffers"] = grad_buffers  # 加载时，复制给model.buffers中的grad_data
        
        # 保存powerSGD_state，注意每个rank都有自己的powerSGD_state，文件名不同
        rank_prefix = best1_prefix + f"rank{ddp_rank}_"
        torch.save(powerSGD_state_dict, os.path.join(ckpt_out_dir, rank_prefix + "powerSGD_state.pt"))
        powerSGD_state_dict = None  # 保存完，可以删除引用了
    
    if ddp_local_rank == 0:
        # 3. 保存实验过程日志，可用resplot来展示，放在最后保存，可以以此判断之前的文件是否保存成功
        # 每个节点的local rank0都需要保存下，可以用于判断之前的文件是否保存成功
        # 其中只有master_process的reslog文件才有值，其他的都是空文件（因为reslog.log只在mater_process用）
        reslog.save(os.path.join(ckpt_out_dir, best1_prefix + "reslog.pkl"))
    
    # 记得还原PowerSGD的进程组状态
    if ddp and use_powerSGD_hook:
        # 直接保存process_group有问题：https://discuss.pytorch.org/t/how-to-resume-with-powersgd-enabled-training/148747/2
        powerSGD_state.process_group = pg
        
    # 等待所有rank都保存完
    if ddp:
        torch.distributed.barrier()
