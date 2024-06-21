import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2ForCausalLM


batch_size = 2
max_seq_len = 10
device = "cuda:0"

input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), dtype=torch.int32).to(device)
labels = torch.randint(0, 1000, (batch_size, max_seq_len), dtype=torch.int32).to(device)
attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.int32).to(device)

attention_mask[0][0:3] = 0
attention_mask[0][6:8] = 0

attention_mask[1][0:2] = 0
attention_mask[1][5:8] = 0


model_path = "./my_Qwen2-0.5B"
model = Qwen2ForCausalLM.from_pretrained(model_path, _attn_implementation="flash_attention_2").to(device)

output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

print(1)