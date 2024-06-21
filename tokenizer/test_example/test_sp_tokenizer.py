import sentencepiece as spm
from transformers import AutoTokenizer

from sp_bbpe_tokenizer import MySPTokenizer


save_dir = "./sp_bbpe_tokenizer"
model_file = "./my_sp_bbpe_tokenizer_20G/my_sp_bbpe_tokenizer_20G.model"

tokenizer = MySPTokenizer(model_file=model_file)
# tokenizer = AutoTokenizer.from_pretrained("./sp_bbpe_tokenizer", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("./testtest", trust_remote_code=True)

text = """<s>1<|beginoftext|><|UNK|>"""

ids = tokenizer(text, allowed_special="all", return_tensors='pt')
print(ids)

# tokenizer.add_special_tokens("user_defined_token")

# tokenizer.save_pretrained("./testtest")

print(tokenizer.vocab_size)

# print(tokenizer.special_tokens)

# print(tokenizer.decode(ids, skip_special_tokens=True))

tokenizer.save_pretrained("./sp_bbpe_tokenizer")
