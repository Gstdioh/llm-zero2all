import os

import json

import sentencepiece as spm

import tokenizers
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Split, ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import PreTrainedTokenizerFast

from config import TokenizerConfig


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
def get_training_corpus(all_cropus_file_abspaths: list, file_bytes="2M", max_train_bytes="5G"):
    # 类似data/02_train_data/get_txt_for_tokenizer.py中的处理
    # file_bytes: 一个文件的大小，单位为字节
    file_bytes = convert_mem2num(file_bytes)
    max_train_bytes = convert_mem2num(max_train_bytes)
    
    # 循环处理所有文件
    buffer_data = []  # 一个文件的数据
    sum_bytes = 0
    train_bytes = 0
    for file_path in all_cropus_file_abspaths:
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


def train_hf_tokenizer(vocab_size):
    tokenizerConfig = TokenizerConfig()

    tokenizer = ByteLevelBPETokenizer()  # 会预先构建256的词表

    # 使用自己的正则表达式进行预分词，并且转换为字节级别
    # 注意，空格会转换为"Ġ"，ByteLevelBPETokenizer解码时会将"Ġ"转换为空格
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
        Split(pattern=tokenizers.Regex(tokenizerConfig.MY_SPLIT_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])

    # ByteLevelBPETokenizer的训练
    # 1. 使用文件训练
    # tokenizer.train(
    #     tokenizerConfig.files_for_train_tokenizer,
    #     vocab_size=vocab_size,
    #     show_progress=True,
    # )
    # 2. 使用迭代器训练
    tokenizer.train_from_iterator(
        get_training_corpus(tokenizerConfig.all_cropus_file_abspaths),
        vocab_size=vocab_size,
        show_progress=True,
    )
    
    # 训练完后再添加特殊token
    tokenizer.add_special_tokens(tokenizerConfig.SPECIAL_TOKENS)
    
    tokenizer.save("./tokenizer/my_hf_bbpe_tokenizer.json")
    
    print("finish!")


def train_spm_tokenizer(vocab_size) -> None:
    '''
    使用sentencepiece训练BPE，缺点只能加载300万行，16G内存会OOM
    '''
    tokenizerConfig = TokenizerConfig()
    
    tokenizer = spm.SentencePieceTrainer.train(
        input=tokenizerConfig.all_cropus_file_abspaths, 
        model_prefix='my_spm_bbpe_tokenizer', 
        model_type='bpe',
        vocab_size=vocab_size, 
        user_defined_symbols=tokenizerConfig.SPECIAL_TOKENS,
        input_format="text",
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        # max_sentence_length=1024,
        # shuffle_input_sentence=True,
        character_coverage=0.9995,
        byte_fallback=True,
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )

    # 模型文件保存在my_tokenizer下
    
    
if __name__ == '__main__':

    # train_spm_tokenizer(300)
    train_hf_tokenizer(64000)

    # train_my_huggingface_wiki_tokenizer(cropus_file=cropus_file, token_type='char') # token_type must be 'char' or 'byte'
