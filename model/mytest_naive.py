import torch

from config_naive import Z2allConfig
from model_naive import Z2allModel

from transformers import LlamaForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Z2allConfig(
    vocab_size=32000,
    hidden_dim=256,
    # intermediate_size=11008,  # FFN中间层的大小
    multiple_of=32,
    n_layers=32,
    n_heads=32,
    n_kv_heads=None,  # 用于GQA
    hidden_act="silu",  # FFN中的激活函数
    max_seq_len=15,
    initializer_range=0.02,  # 参数初始化时的标准差
    rms_norm_eps=1e-6,  # 防止除0的小数
    # use_cache=True,  # 是否缓存kv
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,  # 预训练时的张量并行度, tensor parallelism
    tie_word_embeddings=False,  # 是否共享word embedding和word prediction的参数
    rope_theta=10000.0,
    rope_scaling=None,  # 缩放方法，用于长度外推
    attention_bias=False,  # attention中的project是否加bias
    attention_dropout=0.0,
)

model = Z2allModel(config).to(device)

input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 10), dtype=torch.int).to(device)

model.eval()
output_ids = model.generate(ids=input_ids, max_new_tokens=10)

print(1)
