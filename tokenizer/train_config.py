import random
from datetime import datetime

from utils import convert_mem2num, get_training_iterator, get_file_paths


TrainTokenizerConfig_HELP_MESSAGE = """help_message:
        self.vocab_size = 64000  
        # int. 词汇表大小
        self.train_method = "hf"  
        # str: (hf, sp). 训练tokenizer的工具
        self.train_data_dir = "./data/02_train_data"  
        # str. tokenizer训练集，可以包含json和txt，通过后续代码进行区分
        self.file_type = "txt"  
        # str: (txt, json). 使用txt文件训练，还是json文件进行迭代训练
        self.train_size = "3G"  
        # str. 训练集的大小，txt文件下最小单位为200M，，如果是json文件迭代训练下，可以训练任意大小的数据集，单位不限
        self.save_dir = "./tokenizer"  
        # str. 保存tokenizer的目录
        self.tokenizer_prefix = "my_hf_bbpe_tokenizer"  
        # str. tokenizer文件名前缀，也是tokenizer的目录名"""


# ===================================================================================
# 以下为TrainTokenizer的配置，dataclass表明会代码会先进入这个数据类进行一次初始化，然后实例的时候直接构建
class TrainTokenizerConfig:
    """
        txt使用文件形式训练，json使用迭代形式训练
        hf:  txt, json
        sp: txt
        
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
        self.train_data_dir = "./data/02_train_data"  # str. tokenizer训练集，可以包含json和txt，通过后续代码进行区分
        self.train_method = "hf"  # str: (hf, sp). 训练tokenizer的工具
        self.vocab_size = 64000  # int. 词汇表大小
        self.file_type = "txt"  # str: (txt, json). 使用txt文件训练，还是json文件进行迭代训练
        self.train_size = "3G"  # str. 训练集的大小，txt文件下最小单位为200M，，如果是json文件迭代训练下，可以训练任意大小的数据集，单位不限
        self.save_dir = "./tokenizer"  # str. 保存tokenizer的目录
        self.tokenizer_prefix = ""  # str. tokenizer文件名前缀，也是tokenizer的目录名
        #* ===================================================================================
        
        # 判断特殊情况
        # ===================================================================================
        # 防止报错的属性，值无所谓
        self.config_module = "config.my_config"  # 该配置文件的模块位置
        
        # 提示信息
        if kwargs.get('help', False):
            print(TrainTokenizerConfig_HELP_MESSAGE)
            exit()
            
        # 使用kwargs更新self属性，若不存在或类型不符，则报错
        for key, value in kwargs.items():
            if key not in self.__dict__:
                raise ValueError(f"不存在关键字：{key}！")
            if type(value) != type(getattr(self, key)):
                raise ValueError(f"关键字：{key} 的值的类型不符！要求{type(value)}，但是给定值为{type(getattr(self, key))}")
            setattr(self, key, value)
        
        # 若没有指定，则自动设置
        if self.tokenizer_prefix == "":
            self.tokenizer_prefix = f"{self.train_method}_bbpe_tokenizer_vocab{self.vocab_size}_{self.file_type}_{self.train_size}"
            
        # sp工具不能使用json文件
        if self.train_method == "sp" and self.file_type == "json":
            raise ValueError("sp工具不能使用json文件！")
        
        # 文件类型只能是txt或者json
        if self.file_type not in ["txt", "json"]:
            raise ValueError("文件类型只能是txt或者json！")
        
        # 训练工具只能是hf或者sp
        if self.train_method not in ["hf", "sp"]:
            raise ValueError("训练工具只能是hf或者sp！")
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
        
        # 正则表达式，用于hf中的预分词
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
        
        # 12个特殊token
        self.SPECIAL_TOKENS = [
            '<|beginoftext|>',  # 64000
            '<|endoftext|>',    # 64001
            # '<|fim_prefix|>',
            # '<|fim_middle|>',
            # '<|fim_suffix|>',
            '<|endofprompt|>',  # 64002
            '<|im_start|>',     # 64003 input message
            '<|im_end|>',       # 64004
            '<|UNK|>',          # 64005
            '<|PAD|>',          # 64006
            '<|CLS|>',          # 64007
            '<|SEP|>',          # 64008
            '<|MASK|>',         # 64009
            '<|BOS|>',          # 64010
            '<|EOS|>'           # 64011
        ]


# ===================================================================================
# 以下为PreTrain的配置
class PreTrainConfig:
    def __init__(self, **kwargs):
        #* 以下为需要设置的属性
        #* ===================================================================================
        # I/O
        self.out_dir = "out"
        self.eval_interval = 2000
        self.log_interval = 1
        self.eval_iters = 100
        self.eval_only = False  # if True, script exits right after the first eval
        self.always_save_checkpoint = False  # if True, always save a checkpoint after each eval
        self.init_from = "scratch"  # 'scratch' or 'resume'
        
        # wandb logging
        self.wandb_log = False  # disabled by default
        self.wandb_project = "llamac"
        self.wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        # data
        self.batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.max_seq_len = 256
        self.vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
        self.vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
        
        # adamw optimizer
        self.gradient_accumulation_steps = 4  # used to simulate larger batch sizes
        self.learning_rate = 5e-4  # max learning rate
        self.max_iters = 100000  # total number of training iterations
        self.weight_decay = 1e-1
        self.beta1 = 0.9
        self.beta2 = 0.95
        self.grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
        
        # learning rate decay settings
        self.decay_lr = True  # whether to decay the learning rate
        self.warmup_iters = 1000  # how many steps to warm up for
        
        # system
        self.device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        self.dtype = "bfloat16"  # float32|bfloat16|float16
        self.compile = True  # use PyTorch 2.0 to compile the model to be faster
        #* ===================================================================================