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
        self.system_text = "{im_start}{system_name}{role_end}{system_message}{im_end}".format_map(self.conver_style)
        
        if self.tokenizer is not None:
            # 非空，说明需要对输入进行分词
            # 这里对self.conver_style进行pretokenize
            self.tokenized_conver_style = {}
            for key, value in self.conver_style.items():
                self.tokenized_conver_style[key] = self.tokenizer(value, allowed_special="all").input_ids
            # 预先构建好system文本的ids
            self.system_ids = []
            for key in ["im_start", "system_name", "role_end", "system_message", "im_end"]:
                self.system_ids.extend(self.tokenized_conver_style[key])
    
    def get_conversations_text(self, add_system=True) -> str:
        """
        基于现有的messages构建完整的conversations字符串
        
        {im_start}{system_name}{role_end}{system_message}{im_end}\
        {im_start}{user_name}{role_end}user_message{im_end}\
        {im_start}{assistant_name}{role_end}assistant_message{im_end}\
        """
        ret = ""
        if add_system:
            ret = ret + self.system_text
        for role, message in self.messages:
            ret += "{im_start}{role}{role_end}{message}{im_end}".format_map({
                "im_start": self.conver_style["im_start"],
                "role": role,
                "role_end": self.conver_style["role_end"],
                "message": message,
                "im_end": self.conver_style["im_end"]
            })
        
        return ret

    def get_tokenized_conversations(self, return_labels=True, return_type="pt", ignore_index=-100, add_begin=True, add_system=True, user_first=True):
        """
        将messages中的所有消息转换为input_ids，即将conversations拼接，并进行tokenize
        
        这里的tokenize和直接对拼接的conversations进行tokenize不同
        这里会对conversations中的每个部分单独进行tokenize，防止不同的部分tokenize在一块了
        
        return_labels: bool, 是否返回labels，会将labels中的pad部分掩码设置为ignore_index=-100
        
        return_type: str, 返回的类型，"pt"返回torch.tensor, "np"返回np.array, "list"返回list
        
        add_begin, add_system, 默认添加begin_token_id和system_ids
        
        user_first, conversation中第一个角色是否是user
        """
        assert self.tokenizer is not None, "tokenizer is None"
        
        input_ids = []
        if add_begin:
            input_ids = input_ids + [self.tokenizer.begin_token_id]
        if add_system:
            input_ids.extend(self.system_ids)
        
        # 记录需要掩码的部分，即找到assistant_message的第一个id和最后一个id的后一个id的位置，前一个id需要掩码，后一个id不需要掩码
        # 如有7个id：before_messages[0], before_messages[1], messages[0], messages[1], messages[2], after_messages[0], after_messages[1]
        # 不想掩码的部分为 messages[0], messages[1], messages[2], after_messages[0]
        # 左闭右开，即有mask_index [0, 2), [6, 7) 的位置需要掩码
        # 记录 messages[0] 和 messages[2] 的位置
        # 结果 mask_start_index = messages[2].offset + 2 (需要在后面的后面), mask_end_index = messages[0].offset
        # 左闭右开 [mask_start_index, mask_end_index)
        mask_index = []  # [(mask_start_index, mask_end_index), ...]
        mask_start_index = 0
        mask_end_index = 0
        for i, (role, message) in enumerate(self.messages, start=0 if user_first else 1):
            # user_first=True
            # i%2==0: user, i%2==1: assistant
            # user_first=False
            # i%2==0: assistant, i%2==1: user
            
            cur_ids_len = len(input_ids)
            
            before_message_ids = self.tokenized_conver_style["im_start"] + \
                self.tokenized_conver_style["roles_name"][i % 2] + \
                self.tokenized_conver_style["role_end"]
            
            if i % 2 == 1:
                # assistant_message的第一个id位置得到了，则记录一个mask部分
                cur_ids_len += len(before_message_ids)
                mask_end_index = cur_ids_len  # 对应 messages[0].offset
                mask_index.append((mask_start_index, mask_end_index))
                
            message_ids = self.tokenizer(message, allowed_special="all").input_ids
            
            if i % 2 == 1:
                # assistant_message的后一个id位置得到了，则更新mask_start_index
                cur_ids_len += len(message_ids)  # 最后一个id位置
                mask_start_index = cur_ids_len + 2  # 对应 messages[2].offset + 2
            
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
                    labels[mask_start_index: mask_end_index] = [ignore_index] * (mask_end_index - mask_start_index)
            else:
                raise ValueError(f"Unknown return_type: {return_type}")

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
    