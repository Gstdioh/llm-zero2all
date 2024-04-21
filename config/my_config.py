from dataclasses import dataclass
import os
import glob
from os.path import dirname, abspath
import random


def get_file_abspaths(file_dir: str) -> list:
    '''
    通过正则获取某文件夹下的所有文件的绝对路径
    '''
    # random.seed(42)
    file_abspaths = sorted(glob.glob(os.path.join(file_dir, "*")))
    # random.shuffle(file_abspaths)
    return file_abspaths

# ===================================================================================
# 以下为Tokenizer的配置，dataclass表明会代码会先进入这个数据类进行一次初始化，然后实例的时候直接构建
@dataclass
class TokenizerConfig:
    # 每个json文件大小为200MB，可以选择使用多少个文件来训练Tokenizer
    # 共22GB左右的数据
    # baike2018qa,  github-python,  news2016zh,     webtext2019zh,  wikipedia_cn_20240416,  wikipedia_en_20220301
    # 1.6G,         2.2G,           4.0G,           4.0G,           2.7G,                   7.9G
    # 8,            11,             20,             21,             14,                     40  (见get_train_data_json.sh)
    train_data_dir = "/home/guoliuyang/code/03_LLM/llm-zero2all/data/02_train_data"
    cropus_info = [
        {"from": "baike2018qa", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"baike2018qa"))[:]},
        {"from": "github-python", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"github-python"))[:]},
        {"from": "news2016zh", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"news2016zh"))[:]},
        {"from": "webtext2019zh", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"webtext2019zh"))[:]},
        {"from": "wikipedia_cn_20240416", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"wikipedia_cn_20240416"))[:]},
        {"from": "wikipedia_en_20220301", 
         "file_abspaths": get_file_abspaths(os.path.join(train_data_dir, f"wikipedia_en_20220301"))[:]},
        # {"from": "others",
        #  "file_abspaths": ["/home/guoliuyang/code/03_LLM/llm-zero2all/tokenizer/test.txt"]}
    ]
    all_cropus_file_abspaths = []
    for cropus in cropus_info:
        all_cropus_file_abspaths += cropus["file_abspaths"]
    random.seed(42)
    random.shuffle(all_cropus_file_abspaths)
    
    # 使用txt文件训练tokenizer，占用内存大
    # 只取了json一半的数据
    # dir_for_train_tokenizer = "/home/guoliuyang/code/03_LLM/llm-zero2all/data/02_train_data/00_txt_for_train_tokenizer"
    # files_for_train_tokenizer = sorted(glob.glob(os.path.join(dir_for_train_tokenizer, "*")))
    
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
    
    SPECIAL_TOKENS = [
        '<|endoftext|>',
        '<|fim_prefix|>',
        '<|fim_middle|>',
        '<|fim_suffix|>',
        '<|endofprompt|>',
    ]
    