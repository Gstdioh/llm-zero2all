import tokenizers
from tokenizers import SentencePieceBPETokenizer, ByteLevelBPETokenizer, Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.trainers import BpeTrainer

from config import TokenizerConfig

def train_tokenizer():
    vocab_size = 400

    tokenizerConfig = TokenizerConfig()

    # tokenizer = Tokenizer(BPE())  # 不会预先构建，也就是说可能没有256个基础词
    tokenizer = ByteLevelBPETokenizer()  # 会预先构建256的词表

    # 用兼容等价分解合并对utf编码进行等价组合，比如全角A转换为半角A
    # tokenizer.normalizer = tokenizers.normalizers.Sequence([NFKC()])

    # 使用自己的正则表达式进行预分词，并且转换为字节级别
    # 注意，空格会转换为"Ġ"，ByteLevelBPETokenizer解码时会将"Ġ"转换为空格
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        Split(pattern=tokenizers.Regex(tokenizerConfig.MY_SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevelBPETokenizer有
    # tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)

    # ByteLevelBPETokenizer有
    # tokenizer.decoder = tokenizers.decoders.ByteLevel()

    # trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True)
    tokenizer.train(
        ["/home/guoliuyang/code/03_LLM/llm-zero2all/data/baike2018qa/txt_baike2018qa/txt_baike2018qa_0000.txt"],
        vocab_size=vocab_size,
        show_progress=True,
    )

    tokenizer.save("./new_bbpe_mypat_noNFKC.json")

def test_tokenizer():
    tokenizer = Tokenizer.from_file("./bbpe1.json")

    tmp = """  hello world    !   test。好的"""

    res_en = tokenizer.encode(tmp).tokens
    res_de = tokenizer.decode(tokenizer.encode(tmp).ids)
    
    print(res_en)
    print(res_de)


train_tokenizer()
