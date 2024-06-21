import torch
from torch.nn.utils.rnn import pad_sequence
from model import Z2allForCausalLM, Z2allConfig


batch_size = 2
max_seq_len = 10
device = "cuda:0"

input_ids = torch.randint(0, 1000, (batch_size, max_seq_len), dtype=torch.long).to(device)
labels = torch.randint(0, 1000, (batch_size, max_seq_len), dtype=torch.long).to(device)
attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long).to(device)

attention_mask[0][0:3] = 0
attention_mask[0][6:8] = 0

attention_mask[1][0:2] = 0
attention_mask[1][5:8] = 0

model_config = Z2allConfig(
    n_layers=2,
    use_flash=False
)

model = Z2allForCausalLM(model_config).to(device).bfloat16()

output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

print(1)