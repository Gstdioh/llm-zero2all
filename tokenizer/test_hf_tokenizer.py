import sys
from pathlib import Path

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

sys.path.append("../")

from utils import clean_text

# sys.path.append(str(Path(__file__).parent.parent))


# tokenizer = Tokenizer.from_file("my_hf_bbpe_tokenizer.json")

# tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

tokenizer = AutoTokenizer.from_pretrained("my_hf_bbpe_tokenizer")

# tokenizer.save_pretrained("my_hf_bbpe_tokenizer")

text = "我<|endoftext|><|UNK|>d"

en = tokenizer(text)

print(en)

# print(en.tokens)
# print(en.ids)

# print(tokenizer.decode(en.ids))
