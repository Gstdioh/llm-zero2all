import os
import regex as re
from typing import Dict, Optional, Tuple, Union, List
import copy

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast


TOKENIZER_FILE = "tokenizer.json"

# 手动设置特殊token
SPECIAL_START_ID = 64000
SPECIAL_TOKENS = { token: id
    for id, token in enumerate(
        (
            (
                "<|beginoftext|>",  # 64000
                "<|endoftext|>",    # 64001
                "<|endofprompt|>",  # 64002
                "<|im_start|>",     # 64003
                "<|im_end|>",       # 64004
                "<|UNK|>",          # 64005
                "<|PAD|>",          # 64006
                "<|CLS|>",          # 64007
                "<|SEP|>",          # 64008  
                "<|MASK|>",         # 64009
                "<|BOS|>",          # 64010
                "<|EOS|>",          # 64011
            )
        ),
        start=SPECIAL_START_ID,
    )
}


class MyHFTokenizer(PreTrainedTokenizer):
    def __init__(self, tokenizer_file, **kwargs):
        self._auto_map = { "AutoTokenizer": ["hf_bbpe_tokenizer.MyHFTokenizer", None] }  # 添加映射，保证AutoTokenizer.from_pretrained()可以加载
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        
        # 特殊token
        # self.special_tokens = {key: value for key, value in self.tokenizer.added_tokens_encoder.items() if self._is_special_format(key)}
        # 改为手动设置特殊token
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = copy.deepcopy(self.tokenizer.get_vocab())
        self.vocab.update(self.special_tokens)
        self.vocab_r = {v: k for k, v in self.vocab.items()}
        
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.vocab)
        
    def get_vocab(self) -> Dict[str, int]:
        """
        返回词汇表
        """
        return self.vocab
    
    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.tokenizer.vocab_size
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        """
        保存tokenizer所需要的文件
        """
        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
        )
        self.tokenizer.backend_tokenizer.save(tokenizer_file)
        
        return (tokenizer_file,)
        
    # 清洗文本
    def _clean_text(self, text):
        return text.strip()
    
    # 判断是否是特殊token
    def _is_special_format(self, s):
        return bool(re.match(r'<\|.*?\|>', s))
    
    # 调用父类的__call__方法，需要实现下面方法，其可以对batch进行encode
    # ==============================================================================================
    # 1. 将text文本分词为tokens
    def tokenize(self, text, add_begin=False, allowed_special="none", clean_text=True, **kwargs):
        # 将encode改为tokenize
        
        if clean_text:
            text = self._clean_text(text)
        
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        tokens = []
        if not special:
            tokens = [self.vocab_r[id] for id in self.tokenizer.encode(text)]
        else:
            special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
            special_chunks = re.split(special_pattern, text)
            for part in special_chunks:
                if part in special:
                    tokens.append(part)
                else:
                    tokens.extend([self.vocab_r[id] for id in self.tokenizer.encode(part)])
                
        if add_begin:
            tokens = ["<|beginoftext|>"] + tokens
            
        return tokens
    
    def _convert_id_to_token(self, id):
        if id in self.vocab_r:
            return self.vocab_r[id]
        raise ValueError("unknown ids")
    
    def _convert_token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        raise ValueError("unknown token")
    # ==============================================================================================
    
    def convert_id_to_token(self, id):
        if id in self.vocab_r:
            return self.vocab_r[id]
        raise ValueError("unknown ids")
    
    def convert_token_to_id(self, token):
        if token in self.vocab:
            return self.vocab[token]
        raise ValueError("unknown token")
    
    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)
        
    # 自己的encode，功能与不使用batch的__call__一样
    # 目前只能对单个文本进行encode
    def encode(self, text, add_begin=False, allowed_special="none", clean_text=True):
        if clean_text:
            text = self._clean_text(text)
        
        # 处理特殊token，见minbpe项目代码
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        
        token_ids = []
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            token_ids = self.tokenizer.encode(text)
        else:
            # otherwise, we have to be careful with potential special tokens in text
            # we handle special tokens by splitting the text
            # based on the occurrence of any exact match with any of the special tokens
            # we can use re.split for this. note that surrounding the pattern with ()
            # makes it into a capturing group, so the special tokens will be included
            special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
            special_chunks = re.split(special_pattern, text)
            # now all the special characters are separated from the rest of the text
            # all chunks of text are encoded separately, then results are joined
            for part in special_chunks:
                if part in special:
                    # this is a special token, encode it separately as a special case
                    token_ids.append(special[part])
                else:
                    # this is an ordinary sequence, encode it normally
                    token_ids.extend(self.tokenizer.encode(part))
                
        if add_begin:
            token_ids = [self.special_tokens["<|beginoftext|>"]] + token_ids
            
        return token_ids
    
    
    # 调用父类的decode方法，需要实现下面方法，其可以对batch进行decode，decode调用了_decode，会进行to_py_obj
    # 注意，只能对单个文本进行decode，不能进行batch的decode
    # ==============================================================================================
    # 1. 实现_decode
    def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        if skip_special_tokens:
            token_ids = [id for id in token_ids if id not in self.special_tokens.values()]
            return self.tokenizer.decode(token_ids)
        
        # 不跳过特殊字符
        # 特殊字符需要单独处理
        # 注意，速度会慢些，因为需要两次遍历，ids一次，self.tokenizer.decode()一次
        result = []
        cur_ids = []
        for id in token_ids:
            if id in self.special_tokens.values():
                if len(cur_ids) != 0:
                    result.append(self.tokenizer.decode(cur_ids))
                    cur_ids = []
                result.append(self.vocab_r[id])
            else:
                cur_ids.append(id)
        if len(cur_ids) != 0:
            result.append(self.tokenizer.decode(cur_ids))
            
        return "".join(result)
    # ==============================================================================================
    