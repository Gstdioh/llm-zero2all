import os
import importlib

import sentencepiece as spm
import tokenizers
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from .train_config import TrainTokenizerConfig
from utils import kwargs_parse


def train_hf_tokenizer(trainTokenizerConfig):
    # 处理要保存的路径
    tokenizer_prefix = trainTokenizerConfig.tokenizer_prefix
    tokenizer_save_dir = os.path.join(trainTokenizerConfig.save_dir, f"{tokenizer_prefix}")
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer_save_path = os.path.join(tokenizer_save_dir, f"{tokenizer_prefix}/{tokenizer_prefix}.json")
    # tokenizer_fast_save_path = trainTokenizerConfig.save_dir + "/my_hf_bbpe_tokenizer"
    
    # 使用hf的ByteLevelBPETokenizer训练，修改了pre_tokenizer，使用自己的正则表达式进行预分词

    tokenizer = ByteLevelBPETokenizer()  # 会预先构建256的词表

    # 使用自己的正则表达式进行预分词，并且转换为字节级别
    # 注意，空格会转换为"Ġ"，ByteLevelBPETokenizer解码时会将"Ġ"转换为空格
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        Split(pattern=tokenizers.Regex(trainTokenizerConfig.MY_SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevelBPETokenizer的训练
    if trainTokenizerConfig.file_type == "txt":
        # 1. 使用文件训练
        tokenizer.train(
            trainTokenizerConfig.files_for_train_tokenizer,
            vocab_size=trainTokenizerConfig.vocab_size,
            show_progress=True,
        )
    elif trainTokenizerConfig.file_type == "json":
        # 2. 使用迭代器训练
        tokenizer.train_from_iterator(
            trainTokenizerConfig.iterator_for_train_tokenizer,
            vocab_size=trainTokenizerConfig.vocab_size,
            show_progress=True,
        )
    else:
        raise ValueError("file_type must be 'txt' or 'json'")
    
    # 不需要添加了，因为bbpe中肯定包含了\t
    # 添加 \t ，因为清洗的时候可能把\t给去掉了。。。这里添加一下
    # if '\t' not in tokenizer.get_vocab():
    #     tokenizer.add_tokens(['\t'])
    # if '\n' not in tokenizer.get_vocab():
    #     tokenizer.add_tokens(['\n'])
    
    # 训练完后再添加特殊token
    tokenizer.add_special_tokens(trainTokenizerConfig.SPECIAL_TOKENS)
    
    tokenizer.save(tokenizer_save_path)
    
    # 这里不保存为FastTokenizer，后续自己来构建
    # 保存FastTokenizer，保存后可以通过AutoTokenizer.from_pretrained("my_hf_bbpe_tokenizer")加载
    # tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    # tokenizer.save_pretrained(tokenizer_fast_save_path)
    
    print(f'tokenizer save in path: {tokenizer_save_path}')
    # print(f'fast tokenizer save in path: {tokenizer_fast_save_path}')

    # print(f"\ntrain tokenizer finished. you can use `AutoTokenizer.from_pretrained('{tokenizer_fast_save_path}')` to load and test your tokenizer.")


def train_sp_tokenizer(trainTokenizerConfig) -> None:
    # 处理要保存的路径
    tokenizer_prefix = trainTokenizerConfig.tokenizer_prefix
    tokenizer_save_dir = os.path.join(trainTokenizerConfig.save_dir, f"{tokenizer_prefix}")
    os.makedirs(tokenizer_save_dir, exist_ok=True)
    tokenizer_save_path = os.path.join(tokenizer_save_dir, f"{tokenizer_prefix}")
    
    # 使用sp训练，会自动保存
    tokenizer = spm.SentencePieceTrainer.train(
        input=trainTokenizerConfig.files_for_train_tokenizer,  # 训练的文件列表或者单个文件
        model_prefix=tokenizer_save_path,  # 保存的目录
        model_type='bpe',
        vocab_size=trainTokenizerConfig.vocab_size,  # 其值应该大于等于字符表的大小
        user_defined_symbols=trainTokenizerConfig.SPECIAL_TOKENS,  # 特殊token
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,  # 分割数字，即将数字分割为单个字符
        allow_whitespace_only_pieces=True,  # 允许空格作为一个token
        # max_sentence_length=1024,
        # shuffle_input_sentence=True,
        character_coverage=0.9995,  # 控制字符表的大小，对于中文这种字符多的语言，设置为0.9995，对于英文，设置为1
        byte_fallback=True,  # 开启后，bpe就等价于bbpe
        unk_surface=r" \342\201\207 ",  # 未知token的表示，为"⁇"
        normalization_rule_name="identity",  # 不进行任何规范化，若不指定，默认为"nmt_nfkc"
        required_chars="\t",  # 必须包含的字符，因为清洗的时候可能把\t给去掉了。。。这里添加一下
    )


def train_tokenizer(trainTokenizerConfig):
    '''
    使用hf和sp训练BPE
    '''
    if trainTokenizerConfig.train_method == "hf":
        train_hf_tokenizer(trainTokenizerConfig)
    elif trainTokenizerConfig.train_method == "sp":
        train_sp_tokenizer(trainTokenizerConfig)
    else:
        raise ValueError("train_method must be 'hf' or 'sp'")


if __name__ == '__main__':
    # 解析命令行参数，包括vocab_size, train_method, file_type，有默认值
    # python train_tokenizer.py --vocab_size 64000 --train_method hf --file_type txt
    # python train_tokenizer.py --vocab_size 64000 --train_method sp --file_type txt
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("-v", "--vocab_size", type=int, required=True, help="vocab size")
    # parser.add_argument("-h", "--help", type=str, default="sp", help="hf or sp")
    # kwargs = vars(parser.parse_args())
    
    # 命令
    # python train_tokenizer.py --train_method hf
    # python train_tokenizer.py --train_method sp --train_size 10G
    
    # 解析命令行参数
    kwargs = kwargs_parse()
    
    # 动态导入配置模块
    if 'config_module' not in kwargs:
        kwargs['config_module'] = 'config.my_config'
    config_module = importlib.import_module(kwargs['config_module'])
    TrainTokenizerConfig = getattr(config_module, 'TrainTokenizerConfig')
    
    trainTokenizerConfig = TrainTokenizerConfig(**kwargs)
    train_tokenizer(trainTokenizerConfig)
    