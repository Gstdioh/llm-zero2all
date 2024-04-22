import os
import glob
from os.path import dirname, abspath
import random
from utils import convert_mem2num, get_training_iterator, get_file_paths


HELP_MESSAGE = """help_message:
        self.vocab_size = 64000  
        # int. 词汇表大小
        self.train_method = "hf"  
        # str: (hf, spm). 训练tokenizer的工具
        self.train_data_dir = "./data/02_train_data"  
        # str. tokenizer训练集，可以包含json和txt，通过后续代码进行区分
        self.file_type = "txt"  
        # str: (txt, json). 使用txt文件训练，还是json文件进行迭代训练
        self.train_size = "3G"  
        # str. 训练集的大小，txt文件下最小单位为200M，，如果是json文件迭代训练下，可以训练任意大小的数据集，单位不限"""


# ===================================================================================
# 以下为TrainTokenizer的配置，dataclass表明会代码会先进入这个数据类进行一次初始化，然后实例的时候直接构建
class TrainTokenizerConfig:
    """
        txt使用文件形式训练，json使用迭代形式训练
        hf:  txt, json
        spm: txt
        
        json中
        每个json文件大小为200MB，可以选择使用多少个文件来训练Tokenizer
        共22GB左右的数据
        baike2018qa,  github-python,  news2016zh,     webtext2019zh,  wikipedia_cn_20240416,  wikipedia_en_20220301
        1.6G,         2.2G,           4.0G,           4.0G,           2.7G,                   7.9G
        8,            11,             20,             21,             14,                     40  (见get_train_data_json.sh)
        txt中即将上述json的文本拼接为txt文件
    """
    def __init__(self, **kwargs):
        #* 以下为需要设置的属性
        #* ===================================================================================
        self.vocab_size = 64000  # int. 词汇表大小
        self.train_method = "hf"  # str: (hf, spm). 训练tokenizer的工具
        self.train_data_dir = "./data/02_train_data"  # str. tokenizer训练集，可以包含json和txt，通过后续代码进行区分
        self.file_type = "txt"  # str: (txt, json). 使用txt文件训练，还是json文件进行迭代训练
        self.train_size = "3G"  # str. 训练集的大小，txt文件下最小单位为200M，，如果是json文件迭代训练下，可以训练任意大小的数据集，单位不限
        self.save_dir = "./tokenizer"  # str. 保存tokenizer的目录
        #* ===================================================================================
        
        # 判断特殊情况
        # ===================================================================================
        # 防止报错的属性，值无所谓
        self.config_module = "config.my_config"  # 该配置文件的模块位置
        
        # 提示信息
        if kwargs.get('help', False):
            print(HELP_MESSAGE)
            exit()
            
        # 使用kwargs更新self属性，若不存在或类型不符，则报错
        for key, value in kwargs.items():
            if key not in self.__dict__:
                raise ValueError(f"不存在关键字：{key}！")
            if type(value) != type(getattr(self, key)):
                raise ValueError(f"关键字：{key} 的值的类型不符！要求{type(value)}，但是给定值为{type(getattr(self, key))}")
            setattr(self, key, value)
            
        # spm工具不能使用json文件
        if self.train_method == "spm" and self.file_type == "json":
            raise ValueError("spm工具不能使用json文件！")
        
        # 文件类型只能是txt或者json
        if self.file_type not in ["txt", "json"]:
            raise ValueError("文件类型只能是txt或者json！")
        
        # 训练工具只能是hf或者spm
        if self.train_method not in ["hf", "spm"]:
            raise ValueError("训练工具只能是hf或者spm！")
        # ===================================================================================
        
        # 获取训练数据，通过递归查找所有文件
        self.files_for_train_tokenizer = get_file_paths(self.train_data_dir, self.file_type)
        random.seed(42)
        random.shuffle(self.files_for_train_tokenizer)  # 打乱训练集，使得各类文件均匀
        # 修改训练集大小
        # txt文件，则选择多少个文件
        # json文件，则限制迭代器训练的大小，可以训练任意大小的数据集
        if self.file_type == "txt":
            self.files_for_train_tokenizer = self.files_for_train_tokenizer[:convert_mem2num(self.train_size) // convert_mem2num("200M")]
        elif self.file_type == "json":
            self.iterator_for_train_tokenizer = get_training_iterator(self.files_for_train_tokenizer, buffer_bytes="2M", max_train_bytes=self.train_size)
        
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
            '<|beginoftext|>',
            '<|endoftext|>',
            # '<|fim_prefix|>',
            # '<|fim_middle|>',
            # '<|fim_suffix|>',
            '<|endofprompt|>',
            '<|im_start|>',  # input message
            '<|im_end|>',
            '<|UNK|>',
            '<|PAD|>',
            '<|CLS|>',
            '<|SEP|>',
            '<|MASK|>',
            '<|BOS|>',
            '<|EOS|>'
        ]
        