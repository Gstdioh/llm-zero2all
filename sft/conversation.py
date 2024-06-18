from typing import List, Tuple, Union
import copy

import numpy as np
import torch


class Conversation:
    """A class that manages prompt templates and keeps all conversation history.

    style of conversation
    
    {im_start}{system_name}{role_end}{system_message}{im_end}\
    {im_start}{user_name}{role_end}user_message{im_end}\
    {im_start}{assistant_name}{role_end}assistant_message{im_end}\
    ...
    """
    im_start = "<|im_start|>"
    role_end = "<|role_end|>"
    im_end = "<|im_end|>"
    
    system_name = "system"
    roles_name = ("user", "assistant")
    
    system_message = "You are a helpful assistant."
    
    tokenizer = None
    
    def __init__(self, conver_style=None, tokenizer=None):
        self.conver_style = conver_style
        if self.conver_style is None:
            self.conver_style = {
                "im_start": "<|im_start|>",
                "role_end": "<|role_end|>",
                "im_end": "<|im_end|>",
                "system_name": "system",
                "roles_name": ("user", "assistant"),
                "system_message": "You are a helpful assistant."
            }
        self.tokenizer = tokenizer

        self.messages: List[List[str, str]] = []
        
        # 预先构建好system文本
        self.system_full_text = "{im_start}{system_name}{role_end}{system_message}{im_end}".format_map(self.conver_style)
        
        if self.tokenizer is not None:
            # 非空，说明需要对输入进行分词
            # 这里对self.conver_style进行pretokenize
            self.tokenized_conver_style = {}
            for key, value in self.conver_style.items():
                self.tokenized_conver_style[key] = self.tokenizer(value, allowed_special="all").input_ids
            # 预先构建好system文本的input_ids
            self.system_full_ids = []
            for key in ["im_start", "system_name", "role_end", "system_message", "im_end"]:
                self.system_full_ids.extend(self.tokenized_conver_style[key])
    
    def get_prompt(self) -> str:
        """
        基于现有的messages构建完整的提示字符串
        
        {im_start}{system_name}{role_end}{system_message}{im_end}\
        {im_start}{user_name}{role_end}user_message{im_end}\
        {im_start}{assistant_name}{role_end}assistant_message{im_end}\
        """
        ret = copy.deepcopy(self.system_full_text)
        for role, message in self.messages:
            ret += "{im_start}{role}{role_end}{message}{im_end}".format_map({
                "im_start": self.conver_style["im_start"],
                "role": role,
                "role_end": self.conver_style["role_end"],
                "message": message,
                "im_end": self.conver_style["im_end"]
            })
        
        return ret

    def get_tokenized_prompt(self, return_labels=True, return_type="pt", ignore_index=-100, add_begin=True):
        """
        将messages中的所有消息转换为input_ids，即将prompt进行tokenize
        
        这里的tokenize和直接对prompt进行tokenize不同
        这里会对prompt中的每个部分单独进行tokenize，防止不同的部分tokenize在一块了
        
        labels: bool, 是否返回labels，会将labels中的pad部分掩码设置为ignore_index=-100
        """
        assert self.tokenizer is not None, "tokenizer is None"
        
        input_ids = [self.tokenizer.begin_token_id] if add_begin else []
        input_ids.extend(self.system_full_ids)
        # 记录需要掩码的部分，即找到assistant_message前一个id和后一个id的位置，前一个id不需要掩码，后一个id需要掩码
        mask_index = []  # [(mask_start_index, mask_end_index), ...]
        mask_start_index = 0
        mask_end_index = 0
        for i, (role, message) in enumerate(self.messages):
            # i%2==0: user, i%2==1: assistant
            
            cur_ids_len = len(input_ids)
            
            before_message_ids = self.tokenized_conver_style["im_start"] + \
                self.tokenized_conver_style["roles_name"][i % 2] + \
                self.tokenized_conver_style["role_end"]
            
            if i % 2 == 1:
                # assistant_message的前一个id位置得到了，则记录一个mask部分
                cur_ids_len += len(before_message_ids)
                mask_end_index = cur_ids_len - 1  # 前一个id位置
                mask_index.append((mask_start_index, mask_end_index))
                
            message_ids = self.tokenizer(message, allowed_special="all").input_ids
            
            if i % 2 == 1:
                # assistant_message的后一个id位置得到了，则更新mask_start_index
                cur_ids_len += len(message_ids)  # 后一个id位置
                mask_start_index = cur_ids_len
            
            after_message_ids = self.tokenized_conver_style["im_end"]
            
            input_ids.extend(before_message_ids + message_ids + after_message_ids)
        
        if return_type == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.int64)
        elif return_type == "np":
            input_ids = np.array(input_ids, dtype=np.int64)
        
        labels = None
        if return_labels:
            labels = copy.deepcopy(input_ids)
            if return_type == "pt" or return_type == "np":
                for (mask_start_index, mask_end_index) in mask_index:
                    labels[mask_start_index: mask_end_index] = ignore_index
            elif return_type == "list":
                for (mask_start_index, mask_end_index) in mask_index:
                    for i in range(mask_start_index, mask_end_index):
                        labels[i] = ignore_index

        return dict(
            input_ids=input_ids,
            labels=labels
        )
        
    def append_message(self, role: str, message: str):
        """Append a new message."""
        if role not in self.roles_name:
            raise ValueError(f"role must be one of {self.roles_name}")
        self.messages.append([role, message])

    def clear(self):
        """Clear all messages."""
        self.messages = []
    