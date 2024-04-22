import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
import tokenizers.pre_tokenizers
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Punctuation, Digits, Metaspace, Split

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

MY_SPLIT_PATTERN_LIST = [
    r"""'(?i:[sdmt]|ll|ve|re)""",       # "'s"
    r"""\s?+[^\r\n\p{N}\s\p{P}\p{S}\u4e00-\u9fa5]+""",   # " a", "a"，除了中文 
    r"""\p{N}{1,3}""",                  # "123"
    r""" ?[^\s\p{L}\p{N}]++[\r\n]*""",  # " ??\n\n", "?"
    r"""\s*[\r\n]""",                   # "  \n\n"
    r"""\s+(?![^\r\n\p{N}\s\p{P}\p{S}\u4e00-\u9fa5])""", # "  a" -> " " 即不匹配字母前的一个空格，除了中文
    r"""\s+""",                         # "  " 匹配尾部的空格
    r"""[\u4e00-\u9fa5]+""",  # 中文分出来 
]
MY_SPLIT_PATTERN = "|".join(MY_SPLIT_PATTERN_LIST)

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
#     [Punctuation(), Digits(individual_digits=True), Metaspace()]
# )

tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
    [Split(tokenizers.Regex(MY_SPLIT_PATTERN), behavior="isolated")]
)

text = """人站在地球上为什么没有头朝下的感觉 地球上重力作用一直是指向球心的，因此
只要头远离球心，人们就回感到头朝上。
我的小baby我的小baby-辛巴。"""

str = tokenizer.pre_tokenizer.pre_tokenize_str(text)

print(str)
