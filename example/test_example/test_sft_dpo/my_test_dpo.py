import os
import copy
import logging
import time
logging.basicConfig(level=logging.INFO)

from transformers import AutoTokenizer
from model import Z2allForCausalLM, Z2allConfig

import torch
from peft import get_peft_model, LoraConfig

from train_task import Task
from utils.train import forward_step
from utils.dpo import DPOConfig, create_reference_model


# -----------------------------------------------------------------------------
# data
train_data_dir = "data/05_dpo_data/DPO-En-Zh-20k"
valid_data_dir = "data/05_dpo_data/DPO-En-Zh-20k"
num_workers = 0  # 数据加载器的工作进程数
skip_scaling_factor = 1.0  # 跳过的数据集数可能需要乘的数，因为非index的pretrain_dataset构造方式有点不同，GPU个数变化的时候需要设置
## global_batch_size = batch_size * gradient_accumulation_steps * ddp_world_size
batch_size = 2  # if gradient_accumulation_steps > 1, this is the micro-batch size
## gradient_accumulation_steps=gradient_accumulation_steps*ddp_world_size
gradient_accumulation_steps = 128  # used to simulate larger batch sizes，后面会//ddp_world_size
max_seq_len = 2048
grad_div_total_tokens = False  # 是否在计算梯度时除以总的token数，设置reduction="none" and grad_scaling_before_comm=False，使用PowerSGD时不能使用（loss不好，可能因为PowerSGD对数大的压缩不好，有正交化操作）
# dataset, task
task_type = "dpo"  # pretrain|sft|dpo
# pretrain, sft
use_dataset_with_index = False  # pretrain和sft下可用，是否使用索引来遍历数据集，需要先通过build_sample_index_map.py构建sample的索引
# sft, dpo
tokenizer_dir = "tokenizer/hf_bbpe_tokenizer"
user_name = "human"
assistant_name = "gpt"
# sft
sft_type = "conversation"  # sft任务的类型，"conversation"（多轮），"instruction"（单轮指令）
# dpo
max_prompt_len = None  # 默认为max_seq_len//2
prompt_truncation_mode = "keep_end"  # prompt保留的方式，"keep_end"（保留后面），"keep_start"（保留前面）
num_cpus = 1  # dpo预处理时使用的进程数，并行处理的进程数，太大可能会卡住，设置为2还是会报错（不清楚为什么。。。）, RuntimeError: unable to mmap 704 bytes from file <filename not specified>: Cannot allocate memory (12)
# -----------------------------------------------------------------------------
# peft
use_peft = True
lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
lora_task_type = "CAUSAL_LM"
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
lora_modules_to_save = ["encoder", "decoder"]
# dpo_config
ignore_index = -100
dpo_loss_type = "sigmoid"
dpo_beta = 0.1
dpo_label_smoothing = 0.0

device = "cpu"

# -----------------------------------------------------------------------------
# 任务构造器，用于生成训练和验证数据，不同进程会有不同的rng种子
# pretrian下已经预先进行了toeknize，所以不需要tokenizer
if task_type == "pretrain":
    tokenizer = None
elif task_type == "sft" or task_type == "dpo":
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
else:
    raise ValueError(f"Unknown task_type: {task_type}")
task_kwargs = dict(
    task_type=task_type,
    batch_size=batch_size,
    device=device,
    num_workers=num_workers,
    # same
    max_seq_len=max_seq_len,
    use_dataset_with_index=use_dataset_with_index,
    # sft, dpo
    tokenizer=tokenizer,
    user_name=user_name,
    assistant_name=assistant_name,
    # sft
    sft_type=sft_type,
    # dpo
    max_prompt_len=max_prompt_len,
    prompt_truncation_mode=prompt_truncation_mode,
    num_cpus=num_cpus,
)
task = {
    "train": Task(data_dir=train_data_dir, **task_kwargs),
}

train_batch_iter = task["train"].iter_batches()
cur_batch = next(train_batch_iter)

print("Getting model...")

model_config = Z2allConfig(
    hidden_dim=64,
    n_heads=4,
    n_kv_heads=4,
    n_layers=2,
    use_flash=(device != "cpu"),
    use_fused_rope=(device != "cpu"),
    use_fused_cross_entropy=(device != "cpu"),
    use_fused_dropout_add_norm=(device != "cpu"),
    use_fused_rmsnorm=(device != "cpu"),
    use_fused_swiglu=(device != "cpu"),
)

model = Z2allForCausalLM(model_config).to(device).bfloat16()

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    # task_type=lora_task_type,
    target_modules=lora_target_modules,
    # modules_to_save=lora_modules_to_save,
)

if use_peft:
    model = get_peft_model(model, peft_config)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# dpo_config
dpo_config = None
if task_type == "dpo":
    if use_peft:
        ref_model = None
    else:
        ref_model = create_reference_model(model).to(device)
    dpo_config = DPOConfig(
        ref_model=ref_model,
        use_peft=use_peft,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=ignore_index,
        loss_type=dpo_loss_type,
        beta=dpo_beta,
        label_smoothing=dpo_label_smoothing,
    )

print("start training...")

steps = 10

for i in range(steps):
    
    elapsed_time = time.time()

    loss = forward_step(model, cur_batch, task_type=task_type, dpo_config=dpo_config)

    # 立即取下一个batch
    cur_batch = next(train_batch_iter)

    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"step {i}: ", loss.item(), f"elapsed_time: {time.time() - elapsed_time:.4f}")

print(1)
