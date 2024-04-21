import os
import glob
from os.path import dirname, abspath
import random


def get_file_abspaths(file_dir: str, file_type:str = "txt") -> list:
    '''
    获取当前文件夹下某种类型的所有文件的绝对路径，要递归遍历所有子文件夹
    '''
    file_abspaths = []
    for root, dirs, files in os.walk(file_dir):  # os.walk能遍历所有的子文件夹，递归遍历
        for file in files:
            if file.endswith(file_type):
                file_abspaths.append(os.path.join(root, file))
    return file_abspaths

# ===================================================================================
# 以下为TrainTokenizer的配置，dataclass表明会代码会先进入这个数据类进行一次初始化，然后实例的时候直接构建
class TrainTokenizerConfig:
    def __init__(self, file_type: str = "txt"):
        # txt使用文件形式训练，json使用迭代形式训练
        # hf:  txt, json
        # spm: txt
        
        # json中
        # 每个json文件大小为200MB，可以选择使用多少个文件来训练Tokenizer
        # 共22GB左右的数据
        # baike2018qa,  github-python,  news2016zh,     webtext2019zh,  wikipedia_cn_20240416,  wikipedia_en_20220301
        # 1.6G,         2.2G,           4.0G,           4.0G,           2.7G,                   7.9G
        # 8,            11,             20,             21,             14,                     40  (见get_train_data_json.sh)
        # txt中即将上述json的文本拼接为txt文件
        
        #* 获取训练数据，通过递归查找所有文件
        # ===================================================================================
        train_data_dir = "./data/02_train_data"
        # ===================================================================================
        self.files_for_train_tokenizer = get_file_abspaths(train_data_dir, file_type)
        random.seed(42)
        random.shuffle(self.files_for_train_tokenizer)
        
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
        self.MY_SPLIT_PATTERN = "|".join(MY_SPLIT_PATTERN_LIST)
        
        self.SPECIAL_TOKENS = [
            '<|endoftext|>',
            '<|fim_prefix|>',
            '<|fim_middle|>',
            '<|fim_suffix|>',
            '<|endofprompt|>',
        ]
        