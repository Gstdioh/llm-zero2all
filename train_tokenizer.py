import os
import argparse

import json

import sentencepiece as spm

import tokenizers
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from config import TrainTokenizerConfig


def convert_mem2num(mem_str: str) -> int:
    '''
    将内存字符串转换为数字
    如 "2M" -> 2 * 1024 * 1024
    '''
    # 转为大写
    mem_str = mem_str.upper()
    if mem_str[-1] == "K":
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str[-1] == "M":
        return int(float(mem_str[:-1]) * 1024 * 1024)
    elif mem_str[-1] == "G":
        return int(float(mem_str[:-1]) * 1024 * 1024 * 1024)
    else:
        raise ValueError("内存字符串格式错误！单位应为K、M、G！")

# 构建迭代器来训练，防止OOM
# 经过训练，在22G左右的数据下，使用迭代器训练一样会卡住，在进度57/185的时候，占用89%内存，然后就卡住了
def get_training_iterator(files_for_train_tokenizer: list, file_bytes="2M", max_train_bytes="5G"):
    # 类似data/02_train_data/get_txt_for_tokenizer.py中的处理
    # file_bytes: 一个文件的大小，单位为字节
    file_bytes = convert_mem2num(file_bytes)
    max_train_bytes = convert_mem2num(max_train_bytes)
    
    # 循环处理所有文件
    buffer_data = []  # 一个文件的数据
    sum_bytes = 0
    train_bytes = 0
    for file_path in files_for_train_tokenizer:
        with open(file_path, "r", encoding="utf-8") as fjson:
            # 读取所有行
            for line in fjson.readlines():
                # 将每一行转换为json格式
                data = json.loads(line)
                
                # 按照不同情况，添加额外的标点符号
                if len(data["title"] + data["desc"]) == 0:
                    s = data["content"]
                else:
                    s = data["title"] + data["desc"] + "\n" + data["content"]
                buffer_data.append(s)
                
                # 计算字节数
                tmp = len(buffer_data[-1].encode("utf-8"))
                sum_bytes += tmp
                train_bytes += tmp
                
                if sum_bytes > file_bytes:
                    yield buffer_data
                    buffer_data = []
                    sum_bytes = 0
                    
                    if train_bytes > max_train_bytes:
                        break
                    
        if train_bytes > max_train_bytes:
            break
                    
    # 记得最后剩余的数据
    if len(buffer_data) > 0:
        yield buffer_data


def train_hf_tokenizer(vocab_size, file_type="txt"):
    # 使用hf的ByteLevelBPETokenizer训练，修改了pre_tokenizer，使用自己的正则表达式进行预分词
    
    trainTokenizerConfig = TrainTokenizerConfig(file_type)

    tokenizer = ByteLevelBPETokenizer()  # 会预先构建256的词表

    # 使用自己的正则表达式进行预分词，并且转换为字节级别
    # 注意，空格会转换为"Ġ"，ByteLevelBPETokenizer解码时会将"Ġ"转换为空格
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        Split(pattern=tokenizers.Regex(trainTokenizerConfig.MY_SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevelBPETokenizer的训练
    if file_type == "txt":
        # 1. 使用文件训练
        tokenizer.train(
            trainTokenizerConfig.files_for_train_tokenizer,
            vocab_size=vocab_size,
            show_progress=True,
        )
    elif file_type == "json":
        # 2. 使用迭代器训练
        tokenizer.train_from_iterator(
            get_training_iterator(trainTokenizerConfig.files_for_train_tokenizer),
            vocab_size=vocab_size,
            show_progress=True,
        )
    else:
        raise ValueError("file_type must be 'txt' or 'json'")
    
    # 训练完后再添加特殊token
    tokenizer.add_special_tokens(trainTokenizerConfig.SPECIAL_TOKENS)
    
    tokenizer.save("./tokenizer/my_hf_bbpe_tokenizer.json")
    
    print("finish!")


def train_spm_tokenizer(vocab_size, file_type="txt") -> None:
    '''
    使用sentencepiece训练BPE，缺点只能加载300万行，16G内存会OOM
    '''
    if file_type != "txt":
        raise ValueError("file_type must be 'txt'")
    
    trainTokenizerConfig = TrainTokenizerConfig(file_type)
    
    tokenizer = spm.SentencePieceTrainer.train(
        input=trainTokenizerConfig.files_for_train_tokenizer,  # 训练的文件列表或者单个文件
        model_prefix='my_spm_bbpe_tokenizer',  # 保存的目录
        model_type='bpe',
        vocab_size=vocab_size,  # 其值应该大于等于字符表的大小
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
        normalization_rule_name="identity"  # 不进行任何规范化
    )


def train_tokenizer(vocab_size, train_method="hf", file_type="txt"):
    '''
    使用hf和spm训练BPE
    '''
    if train_method == "hf":
        train_hf_tokenizer(vocab_size, file_type)
    elif train_method == "spm":
        train_spm_tokenizer(vocab_size, file_type)
    else:
        raise ValueError("train_method must be 'hf' or 'spm'")


if __name__ == '__main__':
    # 解析命令行参数，包括vocab_size, train_method, file_type，有默认值
    # python train_tokenizer.py --vocab_size 64000 --train_method hf --file_type txt
    # python train_tokenizer.py --vocab_size 64000 --train_method spm --file_type txt
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True, help="vocab size")
    parser.add_argument("--train_method", type=str, default="hf", help="hf or spm")
    parser.add_argument("--file_type", type=str, default="txt", help="txt -> train from file, json -> train from iterator")
    args = parser.parse_args()
    
    train_tokenizer(args.vocab_size, args.train_method, args.file_type)
    