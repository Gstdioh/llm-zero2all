from transformers import AutoTokenizer, Qwen2ForCausalLM
from model import Z2allForCausalLM, Z2allConfig
import logging
logging.basicConfig(level=logging.INFO)

import torch
from peft import get_peft_model, LoraConfig

from train_task import Task


task_type = "sft"
sft_type = "conversation"

tokenizer_dir = "./tokenizer/hf_bbpe_tokenizer"

max_seq_len = 128
device = "cpu"
batch_size = 2
num_workers = 0
use_dataset_with_index = False

train_data_dir = "data/04_sft_conversation_data"
# train_data_dir = "data/04_sft_conversation_data/01_bin_for_sft_hf"

# -----------------------------------------------------------------------------
# 任务构造器，用于生成训练和验证数据，不同进程会有不同的rng种子
# pretrian下已经预先进行了toeknize，所以不需要tokenizer
if task_type == "pretrain":
    tokenizer = None
elif task_type == "sft":
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
    # sft
    tokenizer=tokenizer,
    sft_type=sft_type,
)
task = {
    "train": Task(data_dir=train_data_dir, **task_kwargs),
}

train_batch_iter = task["train"].iter_batches()
cur_batch = next(train_batch_iter)

print("Getting model...")

model_config = Z2allConfig(
    hidden_dim=128,
    n_layers=2,
    use_flash=(device != "cpu"),
    use_fused_rope=(device != "cpu"),
    use_fused_cross_entropy=(device != "cpu"),
    use_fused_dropout_add_norm=(device != "cpu"),
    use_fused_rmsnorm=(device != "cpu"),
    use_fused_swiglu=(device != "cpu"),
)

model = Z2allForCausalLM(model_config).to(device).bfloat16()

lora_r = 16
lora_alpha = 16
lora_dropout = 0.05
lora_task_type = "CAUSAL_LM"
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"]
lora_modules_to_save = ["encoder", "decoder"]

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    # task_type=lora_task_type,
    target_modules=lora_target_modules,
    # modules_to_save=lora_modules_to_save,
)

model = get_peft_model(model, peft_config)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("start training...")

steps = 10

for i in range(steps):

    output = model(**cur_batch)

    # 立即取下一个batch
    cur_batch = next(train_batch_iter)

    loss = output["loss"]

    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"step {i}: ", loss.item())

print(1)
