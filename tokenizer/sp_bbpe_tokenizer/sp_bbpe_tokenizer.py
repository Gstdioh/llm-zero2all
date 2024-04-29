import os
import regex as re
from typing import Dict, Optional, Tuple, Union, List
import copy

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


SAVE_FILE_NAME = "sp_bbpe_tokenizer.model"

# 特殊token：0-14，<unk>不是 CONTROL，是 UNKNOW
SPECIAL_TUPLE = ('<unk>', '<s>', '</s>', '<|beginoftext|>', '<|endoftext|>', '<|endofprompt|>', '<|im_start|>', '<|im_end|>', '<|UNK|>', '<|PAD|>', '<|CLS|>', '<|SEP|>', '<|MASK|>', '<|BOS|>', '<|EOS|>')


class MySPTokenizer(PreTrainedTokenizer):
    """
    建议：
    1. encode时, tokenzier(text)，              可选参数 (add_begin=False，allowed_special="none"|"all"|"none_raise"|"set", clean_text=True), 实际是convert_tokens_to_ids(tokenize(text))
             或, tokenizer.encode()，           可选参数 (add_begin=False，allowed_special="none"|"all"|"none_raise"|"set", clean_text=True)
    2. decode时, tokenizer.decode(token_ids)    可选参数 (skip_special_tokens=False)
    """

    # 用于from_pretrained()加载模型
    vocab_files_names = {"model_file": SAVE_FILE_NAME}
    
    def __init__(self, model_file, **kwargs):
        self._auto_map = { "AutoTokenizer": ["sp_bbpe_tokenizer.MySPTokenizer", None] }  # 添加映射，保证AutoTokenizer.from_pretrained()可以加载

        self.sp_model = spm.SentencePieceProcessor(model_file=model_file)
        
        # 特殊token
        self.special_tokens = {}
        self._update_special_tokens()
        self.first_not_special_id = self.get_first_not_special_id()  # 第一个不是control的id，从1开始，0是<unk>，<unk>不是 CONTROL，是 UNKNOW
        
        super().__init__(**kwargs)
    
    # 获取模型的proto结构
    def _get_sp_model_proto(self):
        sp_model_proto = sp_pb2_model.ModelProto()
        sp_model_proto.ParseFromString(self.sp_model.serialized_model_proto())
        return sp_model_proto
    
    # 根据sp_model里的特殊token进行更新
    def _update_special_tokens(self):
        self.special_tokens = {}
        sp_model_proto = self._get_sp_model_proto()
        for piece in sp_model_proto.pieces:
            if piece.type == sp_pb2_model.ModelProto.SentencePiece.CONTROL:
                self.special_tokens[piece.piece] = self.sp_model.PieceToId(piece.piece)
    
    # 添加特殊token
    def add_special_tokens(self, added_special_tokens: Union[str, List[str]]):
        if isinstance(added_special_tokens, str):
            added_special_tokens = [added_special_tokens,]
        
        flag = False
        sp_model_proto = self._get_sp_model_proto()
        for token in added_special_tokens:
            if token not in self.special_tokens:
                self.special_tokens[token] = len(sp_model_proto.pieces)
                sp_model_proto.pieces.add(piece=token, score=0, type=sp_pb2_model.ModelProto.SentencePiece.CONTROL)
                flag = True
        # 只有添加了新的特殊token才更新sp_model，防止进行无用的更新
        if flag:
            self.sp_model.LoadFromSerializedProto(sp_model_proto.SerializeToString())
    
    # 第一个不是control的id，从1开始，0是<unk>，<unk>不是 CONTROL，是 UNKNOW
    def get_first_not_special_id(self):
        sp_model_proto = self._get_sp_model_proto()
        for piece in sp_model_proto.pieces[1:]:  # 从1开始，0是<unk>，<unk>不是 CONTROL，是 UNKNOW
            if piece.type != sp_pb2_model.ModelProto.SentencePiece.CONTROL:
                return self.sp_model.PieceToId(piece.piece)
    
    # 删除特殊token，注意不能删除最前面的特殊token，因为删除了会导致后面所有id都变化
    #! 注意，删除操作必须只能在训练LLM之前使用，因为删除某个token后，可能导致其他token的id变化，会使得tokenizer与LLM不匹配了
    def delete_special_tokens(self, deleted_special_tokens: Union[str, List[str]]):
        if isinstance(deleted_special_tokens, str):
            deleted_special_tokens = [deleted_special_tokens,]
        
        flag = False
        sp_model_proto = self._get_sp_model_proto()
        for token in deleted_special_tokens:
            if token in self.special_tokens:
                if self.sp_model.PieceToId(token) >= self.first_not_special_id:
                    del self.special_tokens[token]
                    sp_model_proto.pieces.pop(self.sp_model.PieceToId(token))
                    flag = True
                else:
                    print(f"删除 {token} 失败，不能删除最前面的默认特殊token！")
        # 只有删除了特殊token才更新sp_model，防止进行无用的更新
        if flag:
            self.sp_model.LoadFromSerializedProto(sp_model_proto.SerializeToString())
            # 注意，删除特殊token后，需要重新更新special_tokens，因为id可能会变
            self._update_special_tokens()
            
    def get_special_tokens(self):
        return self.special_tokens
        
    def __len__(self):
        return self.sp_model.vocab_size()
        
    def get_vocab(self) -> Dict[str, int]:
        """
        返回词汇表
        """
        return {self.sp_model.IdToPiece(i): i for i in range(self.sp_model.vocab_size())}
    
    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.sp_model.vocab_size()
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        保存tokenizer所需要的文件，即
        """
        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + SAVE_FILE_NAME
        )
        
        sp_model_proto = self._get_sp_model_proto()
        with open(tokenizer_file, "wb") as f:
            f.write(sp_model_proto.SerializeToString())
        
        return (tokenizer_file,)
        
    # 清洗文本
    def _clean_text(self, text):
        return text.strip()
    
    # 判断是否是特殊token
    def _is_special_format(self, s):
        return bool(re.match(r'<\|.*?\|>', s))
    
    # 调用父类的__call__方法，需要实现下面方法，其可以对batch进行encode
    # ==============================================================================================
    # 1. 将text文本分词为tokens，PreTrainedTokenizer的_encode_plus方法会调用这个方法
    # 默认会对特殊token进行分割
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
            tokens = self.sp_model.EncodeAsPieces(text)
        else:
            special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
            special_chunks = re.split(special_pattern, text)
            for part in special_chunks:
                if part in special:
                    tokens.append(part)
                else:
                    tokens.extend(self.sp_model.EncodeAsPieces(part))
                
        if add_begin:
            tokens = ["<|beginoftext|>"] + tokens
            
        return tokens
    
    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)
    
    def _convert_id_to_token(self, id):
        return self.sp_model.IdToPiece(id)
    # ==============================================================================================
    
    def convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp_model.IdToPiece(id)
    
    def convert_tokens_to_string(self, tokens):
        return self.sp_model.DecodePieces(tokens)  # sp_model的decode默认会跳过CONTROL类型的token，即特殊token

    # 有了_convert_token_to_id，父类实现了convert_tokens_to_ids
    # def convert_tokens_to_ids(self, tokens):
    #     return [self.sp_model.PieceToId(token) for token in tokens]

    # 有了_convert_id_to_token，父类实现了convert_ids_to_tokens
    # def convert_tokens_to_ids(self, tokens):
    #     return [self.sp_model.PieceToId(token) for token in tokens]
    
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
            token_ids = self.sp_model.EncodeAsIds(text)
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
                    token_ids.extend(self.sp_model.EncodeAsIds(part))
                
        if add_begin:
            token_ids = [self.special_tokens["<|beginoftext|>"]] + token_ids
            
        return token_ids
    
    
    # 调用父类的decode方法，需要实现下面方法，decode调用了_decode，会进行to_py_obj
    # 注意，只能对单个文本进行decode，不能进行batch的decode
    # ==============================================================================================
    # 1. 实现_decode
    def _decode(self, token_ids: Union[int, List[int]], skip_special_tokens=False, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        if skip_special_tokens:
            return self.sp_model.DecodeIds(token_ids)  # 默认就会跳过CONTROL类型的token，即特殊token
        
        # 不跳过特殊字符
        # 特殊字符需要单独处理
        # 注意，速度会慢些，因为需要两次遍历，ids一次，self.sp_model.decode()一次
        result = []
        cur_ids = []
        for id in token_ids:
            if id in self.special_tokens.values():
                if len(cur_ids) != 0:
                    result.append(self.sp_model.decode(cur_ids))
                    cur_ids = []
                result.append(self.vocab_r[id])
            else:
                cur_ids.append(id)
        if len(cur_ids) != 0:
            result.append(self.sp_model.decode(cur_ids))
            
        return "".join(result)
    # ==============================================================================================
    