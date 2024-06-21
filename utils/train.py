import logging

import torch
import inspect
try:
    from apex.optimizers import FusedAdam
except ImportError:
    FusedAdam = None
from .utils import print_rank0
from .dpo import dpo_forward


logger = logging.getLogger(__name__)


# 配置优化器 (见llama2.c项目)
def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # 所有的权重张量和嵌入张量都会被weight decay，所有的偏置和layernorm都不会被weight decay
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # 统计参数数量
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print_rank0(logger.info, f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print_rank0(logger.info, f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
    if FusedAdam is None:
        # 是否使用fused版本的AdamW优化器
        # Create AdamW optimizer and use the fused version if it is available
        # 检查torch.optim.AdamW函数是否接受一个名为fused的参数，接受说明版本支持fused，速度更快
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print_rank0(logger.info, f"using pytorch fused AdamW: {use_fused}")
    else:
        optimizer = FusedAdam(optim_groups, lr=learning_rate, betas=betas, set_grad_none=True)
        print_rank0(logger.info, f"using apex fused AdamW")

    return optimizer


# 估计模型的flops利用率（MFU, Model FLOPs Utilization），分母为A100 bfloat16峰值FLOPS (见llama2.c项目)
def estimate_mfu(model, fwdbwd_per_iter, dt):
    # fwdbwd_per_iter: 每次迭代的前向和后向传播次数
    # dt: 计算的时间
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    # first estimate the number of flops we do per iteration.
    # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    N = sum(p.numel() for p in model.parameters())
    config = model.config
    L, H, Q, T = config.n_layers, config.n_heads, config.hidden_dim//config.n_heads, config.max_seq_len
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt)  # per second
    flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu


def forward_step(model, batch, task_type="pretrain", dpo_config=None):
    """
    进行一次前向传播，返回loss
    
    DPO下会有不同的行为
    """
    if task_type == "pretrain" or task_type == "sft":
        model_outputs = model(**batch)
        loss = model_outputs["loss"]
    elif task_type == "dpo":
        assert dpo_config is not None, "dpo_config must be provided for DPO task"
        loss, metrics = dpo_forward(model, batch, dpo_config)
    
    return loss
