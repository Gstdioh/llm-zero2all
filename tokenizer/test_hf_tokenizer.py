import sys
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import AutoTokenizer

import torch

from hf_bbpe_tokenizer import MyHFTokenizer

# sys.path.append("../")
# sys.path.append(str(Path(__file__).parent.parent))

save_dir = "./hf_bbpe_tokenizer"
# tokenizer_file = "./my_hf_bbpe_tokenizer_10G/my_hf_bbpe_tokenizer_10G.json"

# tokenizer = MyHFTokenizer(tokenizer_file)
tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

text = "好的åĨĻçļĦæĺ¯燙<|beginoftext|>test    <|UNK|><|endoftext|>"

tokens = tokenizer.tokenize(text, allowed_special="all")
print(tokens)

print(tokenizer.convert_tokens_to_string(tokens))

# ids = tokenizer(text, add_begin=True, allowed_special="all")["input_ids"]
# print(ids)

# print(tokenizer.decode(ids))

# print(tokenizer.tokenize(text))
# print(tokenizer(text, add_begin=True, allowed_special="none")["input_ids"])

# tok = PreTrainedTokenizerFast(tokenizer_file="my_hf_bbpe_tokenizer_10G.json")
# de = tok.decode([123, 312, 122])
# print(de)

# tokenizer.save_pretrained(save_dir)
