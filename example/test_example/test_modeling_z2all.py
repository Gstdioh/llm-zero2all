import torch
from contextlib import nullcontext

from model import Z2allConfig, Z2allModel, Z2allForCausalLM

from transformers import AutoTokenizer

from transformers import AutoConfig, AutoModel


device_type = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer
# ===============================================================================================
# tokenizer = AutoTokenizer.from_pretrained("../tokenizer/hf_bbpe_tokenizer", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/sp_bbpe_tokenizer", trust_remote_code=True)

# text = ["Hello, my dog is cute. 你好吗", "哈哈，还行吧"]

# ids = tokenizer(text)["input_ids"]
# print(ids)

# de = tokenizer.decode(ids[0])
# print(de)
# ===============================================================================================

# config
# ===============================================================================================
# config = AutoConfig.from_pretrained("./hf_z2all_config", trust_remote_code=True)

config = Z2allConfig(
    vocab_size=64200,  # 实际是64012个，写大点方便扩展
    hidden_dim=256,
    # intermediate_size=11008,  # FFN中间层的大小
    multiple_of=32,
    n_layers=3,
    n_heads=32,
    n_kv_heads=8,  # 用于GQA
    max_seq_len=15,
    initializer_range=0.02,  # 参数初始化时的标准差
    rms_norm_eps=1e-6,  # 防止除0的小数
    pad_token_id=None,
    tie_word_embeddings=False,  # 是否共享word embedding和word prediction的参数
    rope_theta=10000.0,
    rope_scaling=None,  # 缩放方法，用于长度外推
    attention_bias=False,  # attention中的project是否加bias
    attention_dropout=0.1,
    dropout1=0.1,
    dropout2=0.1,
    residual_in_fp32=True,  # 残差连接是否使用fp32
)

# config.save_pretrained("./hf_z2all_config")

# ===============================================================================================

model = Z2allForCausalLM(config).to(device_type)

# input_ids (2, 10)
input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 5), dtype=torch.int).to(device_type)

dtype = "bfloat16"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    # else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # 原来的代码
    else torch.autocast(device_type=device_type[:4], dtype=ptdtype)
)

with ctx:
    output = model(input_ids)

# with ctx:
#     model.eval()
#     output_ids = model.generate(ids=input_ids, max_new_tokens=10)
    
#     print(input_ids[0].tolist())
#     print(output_ids[0].tolist())
#     print("-------------------------------")
#     print(tokenizer.decode(input_ids[0]))
#     print(tokenizer.decode(output_ids[0]))

print(1)
