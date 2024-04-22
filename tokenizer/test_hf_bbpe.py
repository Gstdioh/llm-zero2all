import tokenizers
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file("./my_hf_bbpe_tokenizer.json")

print(tokenizer.encode("Hello, y'all! How are you 😁? 你好").tokens)
ids = tokenizer.encode("Hello, y'all! How are you 😁? 你好").ids
print(ids)

print("decode")
print(tokenizer.decode(ids))
