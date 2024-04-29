"""
参考llama2模型结构

注意：
1）注意在RMS、RoPE、attention中的softmax计算时，使用的是float32，并且计算完要转换回原来的类型（如bfloat16）
https://zhuanlan.zhihu.com/p/651588659
2）无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中，这是由优化器更新的内容。
16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。因此，不会发生精度损失。
https://www.zhihu.com/question/616600181/answer/3160547952?utm_source=zhihu
3）使用drop_path，见Megatron-Deepspeed中的transformer.py

优化：
1）融合算子：
    a）flash_attn中的flash attention2
    b）xformers中的fused swiglu
    c）flash_attn中的fused dropout_add_rms_norm, 需要安装 dropout_layer_norm 包，pytorch2.1.0下安装有问题，版本不兼容
        https://github.com/open-mmlab/mmcv/issues/2938
        all dimensions divisible by 8, up to 8192
    d）cd ../csrc/rotary && pip install .
    e）cd ../csrc/xentropy && pip install .
2）其他优化：
    a）将运算中的hidden_state维度从 (bsz, seq_len, hidden_dim) 转换为 (seq_len, bsz, hidden_dim)
       因为通常s远大于b，转换后变成了s个矩阵相乘，这意味着GPU可以同时处理s个矩阵相乘，而不仅仅是b个矩阵相乘，
       将数值大的放在第一维度，使得GPU可以进行更大规模的并行计算，提高计算效率。
"""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from einops import rearrange, repeat

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
except ImportError:
    dropout_add_rms_norm = None
try:
    from xformers.ops import swiglu
except ImportError:
    swiglu = None
try:
    from fused_rotary_embedding import fused_rotary_emb
except ImportError:
    fused_rotary_emb = None
try:
    from fused_cross_entropy import fused_cross_entropy
except ImportError:
    fused_cross_entropy = None

from configuration_z2all import Z2allConfig


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        # 交替放，不能用cat
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    见 flash_attn/layers/rotary.py
    
    x: (seqlen, batch_size, nheads, headdim)
    cos, sin: (seqlen, 1, 1, rotary_dim / 2)  通常来说，rotary_dim = headdim
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... (2 d)" if not interleaved else "... d -> ... (d 2)")
    sin = repeat(sin, "... d -> ... (2 d)" if not interleaved else "... d -> ... (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    )


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    只用在主路径中
    """

    def __init__(self, p=0.):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, hidden_state):
        if self.p == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.p
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        # output = hidden_state.div(keep_prob) * random_tensor
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return hidden_state * random_tensor


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
        self.register_parameter("bias", None)

    def forward(self, hidden_states):
        # 计算时用float32
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        
        return self.weight * hidden_states.to(input_dtype)


class Z2allMLP(nn.Module):
    def __init__(self, config: Z2allConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.intermediate_size = config.intermediate_size
        
        if self.intermediate_size is None:
            # 保持参数量和两个全连接的参数量相近
            self.intermediate_size = self.hidden_dim * 4
            self.intermediate_size = int(2 * self.hidden_dim / 3)
            self.multiple_of = config.multiple_of  # 保持intermediate_size是multiple_of的倍数，在张量并行时使用GPTQ有用
            self.intermediate_size = self.multiple_of * ((self.intermediate_size + self.multiple_of - 1) // self.multiple_of)
        
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.act_fn = nn.SiLU()
        self.op = None
        
        self.use_fused_swiglu = (swiglu is not None) and config.use_fused_swiglu
        if not self.use_fused_swiglu:
            print("WARNING: using slow swiglu. fused swiglu requires xformers")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``
        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        # x (seq_len, bsz, hidden_dim)
        
        if self.use_fused_swiglu:
            # fused swiglu
            output = swiglu(x, 
                            self.gate_proj.weight, self.gate_proj.bias, 
                            self.up_proj.weight, self.up_proj.bias, 
                            self.down_proj.weight, self.down_proj.bias, 
                            op=self.op)
        else:
            # navie实现
            output = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return output


class Z2allAttention(nn.Module):
    def __init__(self, config: Z2allConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.head_dim = self.hidden_dim // self.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.max_seq_len = config.max_seq_len
        self.rope_theta = config.rope_theta
        self.rope_interleaved = config.rope_interleaved
        self.is_causal = True

        if (self.head_dim * self.n_heads) != self.hidden_dim:
            raise ValueError(
                f"hidden_dim 必须能被 n_heads 整除，但是 `hidden_dim`:{self.hidden_dim} 和 `n_heads`:{self.n_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_dim, self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.n_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=config.attention_bias)
        
        # 使用flash attention或者手动实现（见llama.c项目）
        self.use_flash = (flash_attn_func is not None) and config.use_flash
        if not self.use_flash:
            print("WARNING: using slow attention. Flash Attention2 requires flash_attn>=2.0.0")
            mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # model.to("cuda")时，在buffer中的tensor同样会被移动到cuda
            self.register_buffer("mask", mask)
            
        self.use_fused_rope = (fused_rotary_emb is not None) and config.use_fused_rope
        if not self.use_fused_rope:
            print("WARNING: using slow rotary embedding. fused rotary embedding requires rotary_emb")
        
    def apply_rotary_pos_emb(self, x, cos, sin):
        # 计算时用float32
        # x 可能是 bfloat16 (seq_len, bsz, n_heads, head_dim)
        # cos, sin 是 float32 (seq_len, 1, 1, rotary_dim / 2)  通常来说，rotary_dim = headdim
        input_dtype = x.dtype
        
        output = None
        if self.use_fused_rope:
            output = fused_rotary_emb(x, cos, sin, self.rope_interleaved, inplace=True)
        else:
            output = apply_rotary_emb_torch(x, cos, sin, self.rope_interleaved)
        
        # outpout (seq_len, bsz, n_heads, head_dim)
        return output.to(input_dtype)
        
        # 之前的做法
        # q_len, _, _, head_dim = x.shape
        
        # # x: (q_len, bsz, n_heads, head_dim)
        # # cos (max_seq_len * 2, head_dim) -> (q_len, 1, 1, head_dim) [..., [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]]
        # cos = cos[position_ids, None, None, :]
        # sin = sin[position_ids, None, None, :]

        # # rotate_half_q (q_len, bsz, n_heads, head_dim) [..., [-q1, q0, -q3, q2, ...]]
        # rotate_half_q = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(q)
        # rotate_half_k = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(k)

        # q_embed = (q * cos) + (rotate_half_q * sin)
        # k_embed = (k * cos) + (rotate_half_k * sin)

        # # cos, sin 是 float32，所以需要转回去
        # return q_embed.type_as(q), k_embed.type_as(k)

    def update_key_value_cache(self, past_key_values, key_states, value_states, layer_idx):
        # key_states, value_states: (seq_len, bsz, n_kv_heads, head_dim)

        # 更新 seen_tokens
        if layer_idx == 0:
            past_key_values['seen_tokens'] += key_states.shape[0]  # 注意，维度改了，长度在第0维

        # 更新缓存值
        if len(past_key_values['key_states_layers']) <= layer_idx:
            past_key_values['key_states_layers'].append(key_states)
            past_key_values['value_states_layers'].append(value_states)
        else:
            past_key_values['key_states_layers'][layer_idx] = torch.cat([past_key_values['key_states_layers'][layer_idx], key_states], dim=0)
            past_key_values['value_states_layers'][layer_idx] = torch.cat([past_key_values['value_states_layers'][layer_idx], value_states], dim=0)

        return past_key_values

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效操作。
        隐藏状态从 (seq_len, batch, n_kv_heads, head_dim) 变为 (seq_len, batch, n_heads, head_dim)
        
        注意，在使用flash_attn时，不需要手动复制
        """
        seq_len, bsz, n_kv_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(seq_len, bsz, n_kv_heads, n_rep, head_dim)
        return hidden_states.reshape(seq_len, bsz, n_kv_heads * n_rep, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # hidden_states: (q_len, bsz, hidden_dim)
        q_len, bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (q_len, bsz, hidden_dim) -> (q_len, bsz, n_heads, head_dim)
        query_states = query_states.view(q_len, bsz, self.n_heads, self.head_dim)
        key_states = key_states.view(q_len, bsz, self.n_kv_heads, self.head_dim)
        value_states = value_states.view(q_len, bsz, self.n_kv_heads, self.head_dim)

        # q (q_len, bsz, n_heads, head_dim)
        # k (q_len, bsz, n_kv_heads, head_dim)
        # cos, sin (q_len, 1, 1, head_dim / 2)
        query_states = self.apply_rotary_pos_emb(query_states, cos, sin)
        key_states = self.apply_rotary_pos_emb(key_states, cos, sin)

        # 使用kv缓存
        if use_cache and past_key_values is not None:
            past_key_values = self.update_key_value_cache(past_key_values, key_states, value_states, self.layer_idx)
            # k, v (all_seq_len, bsz, n_kv_heads, head_dim)
            key_states = past_key_values['key_states_layers'][self.layer_idx]
            value_states = past_key_values['value_states_layers'][self.layer_idx]

        # flash_attn中可以自动处理分组，所以不需要手动复制
        # GQA，分组查询注意力
        # key_states (all_seq_len, bsz, n_kv_heads, head_dim) -> (all_seq_len, bsz, n_heads, head_dim)
        if not self.use_flash:
            key_states = self.repeat_kv(key_states, self.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.num_key_value_groups)

        # q, k, v (seq_len, bsz, n_heads, head_dim)
        # flash implementation
        if self.use_flash:
            # flash_attn中为 (bsz, seq_len, n_heads, head_dim), 使用这个格式
            #     torch中为  (bsz, n_heads, seq_len, head_dim)
            
            # q, k, v (seq_len, bsz, n_heads, head_dim) -> (bsz, seq_len, n_heads, head_dim)
            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)
            # attn_output (bsz, seq_len, n_heads, head_dim)
            attn_output = flash_attn_func(query_states, key_states, value_states, dropout_p=self.attention_dropout if self.training else 0.0, causal=True)
            # attn_output (seq_len, bsz, n_heads, head_dim)
            attn_output = attn_output.transpose(0, 1)
        else:  # 朴素实现
            # q     (q_len, bsz, n_heads, head_dim)
            # k, v  (all_seq_len, bsz, n_heads, head_dim)
            # key_states包含之前缓存的长度
            
            query_states = query_states.permute(1, 2, 0, 3)  # (bsz, n_heads, q_len, head_dim)
            key_states = key_states.permute(1, 2, 3, 0)  # (bsz, n_heads, head_dim, all_seq_len)
            
            # attn_weights (bsz, n_heads, q_len, all_seq_len)
            attn_weights = torch.matmul(query_states, key_states) / math.sqrt(self.head_dim)

            assert hasattr(self, 'mask')
            attn_weights = attn_weights + self.mask[:, :, position_ids, :]   # (bsz, n_heads, q_len, all_seq_len)

            # 注意力计算时用fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            # attn_weights (bsz, n_heads, q_len, all_seq_len)
            # value_states (bsz, n_heads, all_seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, value_states)  # (bsz, n_heads, q_len, head_dim)
            
            attn_output = attn_output.permute(2, 0, 1, 3)  # (q_len, bsz, n_heads, head_dim)

        # attn_output (q_len, bsz, n_heads, head_dim) -> (q_len, bsz, hidden_dim)
        attn_output = attn_output.reshape(q_len, bsz, self.hidden_dim)

        # attn_output (q_len, bsz, hidden_dim) -> (q_len, bsz, hidden_dim)
        attn_output = self.o_proj(attn_output)

        # attn_output (q_len, bsz, hidden_dim)
        return attn_output, past_key_values


class Z2allDecoderLayer(nn.Module):
    """
    Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP
    """
    
    def __init__(self, config: Z2allConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32

        # Dropout -> Add -> LN
        self.dropout1 = nn.Dropout(config.dropout1)
        self.drop_path1 = DropPath(config.drop_path1)
        self.norm1 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # Self Attention
        self.self_attn = Z2allAttention(config=config, layer_idx=layer_idx)
        
        # Dropout -> Add -> LN
        self.dropout2 = nn.Dropout(config.dropout2)
        self.drop_path2 = DropPath(config.drop_path2)
        self.norm2 = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        # MLP
        self.mlp = Z2allMLP(config)

        self.use_fused_dropout_add_norm = (dropout_add_rms_norm is not None) and config.use_fused_dropout_add_norm
        if not self.use_fused_dropout_add_norm:
            print("WARNING: using slow dropout_add_rms_norm. Fused dropout_add_rms_norm requires dropout_layer_norm")
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        past_key_values: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # hidden_states: (seq_len, bsz, hidden_dim)
        
        # Dropout -> Add -> LN
        if not self.use_fused_dropout_add_norm:
            dropped = self.drop_path1(self.dropout1(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual.to(dtype=self.norm1.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if self.drop_path1.p == 0 or not self.training:
                rowscale1 = None
            else:
                rowscale1 = self.drop_path1(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            hidden_states, residual = dropout_add_rms_norm(
                hidden_states,
                residual,
                self.norm1.weight,
                self.norm1.bias,
                self.dropout1.p if self.training else 0.0,
                self.norm1.eps,
                rowscale=rowscale1,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            
        # Self Attention
        hidden_states, past_key_values = self.self_attn(
            hidden_states=hidden_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
            **kwargs,
        )
        
        # Dropout -> Add -> LN
        if not self.use_fused_dropout_add_norm:
            dropped = self.drop_path2(self.dropout2(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if self.drop_path2.p == 0 or not self.training:
                rowscale2 = None
            else:
                rowscale2 = self.drop_path2(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            hidden_states, residual = dropout_add_rms_norm(
                hidden_states,
                residual,
                self.norm2.weight,
                self.norm2.bias,
                self.dropout2.p if self.training else 0.0,
                self.norm2.eps,
                rowscale=rowscale2,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            
        # MLP
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual, past_key_values


def init_key_value_cache():
    return {"key_states_layers": [], "value_states_layers": [], "seen_tokens": 0}


class Z2allPreTrainedModel(PreTrainedModel):
    """
    用于权重的初始化，保存和加载模型
    """
    config_class = Z2allConfig
    base_model_prefix = "transformer"
    # is_parallelizable = False
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["QWenBlock"]
    # _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        # PreTrainedModel的__init__里，第一个参数inputs[0] = config，自动设置了self.config = config
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
                
    def post_init(self):
        # 初始化权重，应用到模型的所有子模块
        self.apply(self._init_weights)
        # 应用特殊的缩放初始化到残差投影，参考GPT-2论文，（见llama2.c项目）
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=self.config.initializer_range/math.sqrt(2 * self.config.n_layers))

    # TODO
    # def _set_gradient_checkpointing(self, module, value=False):
    #     if isinstance(module, QWenModel):
    #         module.gradient_checkpointing = value


class Z2allModel(Z2allPreTrainedModel):
    def __init__(self, config: Z2allConfig):
        super().__init__(config)
        
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.rope_interleaved = config.rope_interleaved
        self.n_layers = config.n_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.residual_in_fp32 = config.residual_in_fp32
        self.data_type = config.data_type

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_dim, self.padding_idx)
        self.layers = nn.ModuleList(
            [Z2allDecoderLayer(config, layer_idx) for layer_idx in range(self.n_layers)]
        )
        
        # Dropout -> Add -> LN
        self.dropout_last = nn.Dropout(config.dropout_last)
        self.drop_path_last = DropPath(config.drop_path_last)
        self.norm_last = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

        self.use_fused_dropout_add_norm = (dropout_add_rms_norm is not None) and config.use_fused_dropout_add_norm
        if not self.use_fused_dropout_add_norm:
            print("WARNING: using slow dropout_add_rms_norm. Fused dropout_add_rms_norm requires dropout_layer_norm")

        self.output = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 是否共享参数
        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.output.weight
        
        # 预处理RoPE中用到的cos和sin
        cos, sin = self.precompute_cos_sin(
            dim=self.hidden_dim // self.n_heads,
            end=self.max_seq_len * 2,  # 注意，end = self.max_seq_len * 2，因为Llama 2模型的token限制为4096。
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        # 模型保存和加载的时候不用保存这两个tensor，注册是方便.to(device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # 弃用，防止其因混合精度转换为bfloat16，导致精度损失
        # # model.to("cuda")时，在buffer中的tensor同样会被移动到cuda
        # self.register_buffer("cos", cos, persistent=False)
        # self.register_buffer("sin", sin, persistent=False)
        
        # 初始化权重
        self.post_init()

    def rotary_embedding(self, dim, position_ids, base=10000, device=None):
        seq_len = torch.max(position_ids) + 1

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

        position_ids_expanded = position_ids[:, None].float().to(device)  # (seq_len, 1)
        inv_freq_expanded = inv_freq[None, :].float().to(device)  # (1, dim  /2)
        freqs = (position_ids_expanded @ inv_freq_expanded)  # (seq_len, dim / 2)
        
        cos, sin = freqs.cos(), freqs.sin()

        # cos, sin (seq_len, dim / 2)
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.data_type]
        return cos.to(ptdtype), sin.to(ptdtype)

    def linear_scaling_rotary_embedding(self, dim, position_ids, base=10000, scaling_factor=1.0, device=None):
        position_ids = position_ids.float() / scaling_factor

        cos, sin = self.rotary_embedding(dim, position_ids, base=base, device=device)

        return cos, sin

    def dynamic_NTK_scaling_rotary_embedding(self, dim, position_ids, base=10000, scaling_factor=1.0, device=None):
        seq_len = torch.max(position_ids) + 1
        # 见 transformers/models/llama/modeling_llama.py
        base = base * (
            (scaling_factor * seq_len / self.max_seq_len) - (scaling_factor - 1)
        ) ** (dim / (dim - 2))

        cos, sin = self.rotary_embedding(dim, position_ids, base=base, device=device)

        return cos, sin

    # 预先计算RoPE中的cos和sin
    def precompute_cos_sin(self, dim, end, base=10000, rope_scaling=None, device=None):
        assert dim % 2 == 0, f"`head_dim` 必须是偶数, 原因在RoPE, 但是 `head_dim`: {dim}"

        # 注意，end = self.max_seq_len * 2，因为Llama 2模型的token限制为4096。
        # 添加这个乘数而不是直接使用4096允许在训练或微调时对token长度进行动态调整。
        position_ids = torch.arange(0, end, dtype=torch.int64)

        if rope_scaling is None:
            cos, sin = self.rotary_embedding(dim, position_ids, base=base, device=device)
        elif rope_scaling["type"] == "linear":
            cos, sin = self.linear_scaling_rotary_embedding(
                dim, position_ids, base=base, scaling_factor=rope_scaling["factor"], device=device
            )
        elif rope_scaling["type"] == "dynamic":
            cos, sin = self.dynamic_NTK_scaling_rotary_embedding(
                dim, position_ids, base=base, scaling_factor=rope_scaling["factor"], device=device
            )
        else:
            raise ValueError(f"Unknown rope_scaling type, {rope_scaling}")

        # (max_seq_len * 2, head_dim / 2)
        return cos, sin
            
    # 弃用，在attention块中自己构建mask
    def update_causal_mask(self, all_seq_len, dtype, device=None):
        # 根据当前的总长度设置掩码矩阵的大小
        mask = torch.full((all_seq_len, all_seq_len), fill_value=torch.finfo(dtype).min)
        # triangle upper，diagonal为1，表示构建一个上三角矩阵（不包括对角线），即将上三角的元素进行掩码
        causal_mask = torch.triu(mask, diagonal=1)
        
        causal_mask = causal_mask.to(dtype=dtype, device=device)
        
        return causal_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache=False,
        past_key_values: Optional[Dict[str, Any]] = None,
    ) -> BaseModelOutputWithPast:
        """模型的前向传播
        kv缓存的格式: {"key_states_layers": [], "value_states_layers": [], "seen_tokens": 0}

        Args:
            input_ids (torch.Tensor): 输入的token id
            labels (torch.Tensor, optional): 预训练时的标签. Defaults to None.
            use_cache (bool): 是否使用kv缓存. Defaults to False.
            past_key_values (_type_, optional): kv缓存. Defaults to None.

        Returns:
            BaseModelOutputWithPast
        """
        _, seq_len = input_ids.size()
        
        start_pos = 0
        # 初始化kv缓存
        if use_cache:
            if past_key_values is None:
                past_key_values = init_key_value_cache()
            else:
                start_pos = past_key_values['seen_tokens']
        
        all_seq_len = start_pos + seq_len  # 包含了之前缓存的token长度
        
        # 得到position_ids
        position_ids = torch.arange(start_pos, all_seq_len, device=input_ids.device, dtype=torch.int64)
        
        # 取相应的位置，将cos, sin放入cuda, float32
        # cos, sin (max_seq_len, head_dim / 2) -> (seq_len, 1, 1, head_dim / 2)
        cos = self.cos[position_ids, None, None, :]
        sin = self.sin[position_ids, None, None, :]
            
        # token嵌入
        # hidden_states (bsz, seq_len, hidden_dim)
        hidden_states = self.embed_tokens(input_ids)
        
        # hidden_states (bsz, seq_len, hidden_dim) -> (seq_len, bsz, hidden_dim)
        # 将数值大的放在第一维度，这样可以提高GPU的并行度
        hidden_states = hidden_states.transpose(0, 1)
        
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual, past_key_values = decoder_layer(
                hidden_states=hidden_states,
                cos=cos,
                sin=sin,
                residual=residual,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
            
        # hidden_states (seq_len, bsz, hidden_dim)
        # 对于prenorn，最后需要再加一个 Dropout -> Add -> LN
        # 注意，最后的不需要residual了
        if not self.use_fused_dropout_add_norm:
            dropped = self.drop_path_last(self.dropout_last(hidden_states))
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm_last(residual.to(dtype=self.norm_last.weight.dtype))
        else:
            if self.drop_path_last.p == 0 or not self.training:
                rowscale_last = None
            else:
                rowscale_last = self.drop_path_last(
                    torch.ones(
                        hidden_states.shape[:-1],
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                )
            hidden_states = dropout_add_rms_norm(
                hidden_states,
                residual,
                self.norm_last.weight,
                self.norm_last.bias,
                self.dropout_last.p if self.training else 0.0,
                self.norm_last.eps,
                rowscale=rowscale_last,
                prenorm=False,  # 最后的不需要residual了
                residual_in_fp32=self.residual_in_fp32,
            )
            
        # hidden_states (seq_len, bsz, hidden_dim) -> (bsz, seq_len, hidden_dim)
        hidden_states = hidden_states.transpose(0, 1)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Z2allForCausalLM(Z2allPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Z2allConfig):
        super().__init__(config)
        
        self.model = Z2allModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self.use_fused_cross_entropy = (fused_cross_entropy is not None) and config.use_fused_cross_entropy
        if not self.use_fused_cross_entropy:
            print("WARNING: using slow fused cross entropy. Fused cross entropy requires xentropy_cuda_lib")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache=True,
        past_key_values: Optional[Dict[str, Any]] = None,
        loss_fn=None,
    ) -> CausalLMOutputWithPast:
        """模型的前向传播
        kv缓存的格式: {"key_states_layers": [], "value_states_layers": [], "seen_tokens": 0}

        Args:
            input_ids (torch.Tensor): 输入的token id
            labels (torch.Tensor, optional): 预训练时的标签. Defaults to None.
            use_cache (bool): 是否使用kv缓存. Defaults to False.
            past_key_values (_type_, optional): kv缓存. Defaults to None.
            loss_fn=None, 用于计算损失的函数

        Returns:
            CausalLMOutputWithPast(loss, logits, past_key_values)
                if labels is None:
                    loss = None,   logits (bsz, vocab_size),         
                    past_key_values {"key_states_layers": [...], "value_states_layers": [...], "seen_tokens": 0}
                else:
                    loss = tensor, logits (bsz, seq_len, vocab_size) 
                    past_key_values = None
        """
        assert labels is None or use_cache is False, "use_cache=True is incompatible with labels"
        
        # last_hidden_state (bsz, seq_len, hidden_dim)
        model_output = self.model(input_ids, use_cache, past_key_values)
        last_hidden_state, past_key_values = model_output.last_hidden_state, model_output.past_key_values
        
        # last_hidden_state (bsz, seq_len, hidden_dim) -> (bsz, hidden_dim)
        # 推理时，只需要最后一个位置的输出，loss为None
        if labels is None:
            last_hidden_state = last_hidden_state[:, -1, :]
        
        # labels is None，    logits (bsz, vocab_size)
        # labels is not None，logits (bsz, seq_len, vocab_size)
        logits = self.lm_head(last_hidden_state)
        
        loss = None
        if labels is not None:
            # 给定标签，计算损失
            # logits (bsz, seq_len, vocab_size)
            # labels (bsz, seq_len)
            if loss_fn is None:
                if self.use_fused_cross_entropy:
                    loss_fn = fused_cross_entropy
                else:
                    loss_fn = F.cross_entropy
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    # 配置优化器 (见llama2.c项目)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 所有的权重张量和嵌入张量都会被weight decay，所有的偏置和layernorm都不会被weight decay
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # 统计参数数量
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")  # 逗号用于分隔千位
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 是否使用fused版本的AdamW优化器
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # 估计模型的flops利用率（MFU），分母为A100 bfloat16峰值FLOPS (见llama2.c项目)
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # fwdbwd_per_iter: 每次迭代的前向和后向传播次数
        # dt: 计算的时间
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        config = self.config
        L, H, Q, T = config.n_layers, config.n_heads, config.hidden_dim//config.n_heads, config.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, ids, max_new_tokens, max_context_len=-1, use_cache=True, temperature=1.0, top_k=None):
        """使用模型生成文本
        1. 是否使用kv缓存, 并且裁剪长度
        2. 前向传播模型
        3. 采样idx_next
        Args:
            ids (torch.Tensor): 输入的token id
            max_new_tokens (int): 生成的最大新token数量
            max_context_len (int, optional): 最大上下文长度, 用于裁剪kv缓存. Defaults to -1.
            use_cache (bool, optional): 是否使用kv缓存. Defaults to True.
            temperature (float, optional): 生成的温度. Defaults to 1.0.
            top_k (int, optional): 只从前k个采样. Defaults to None.
        Returns:
            torch.Tensor: 生成的token id
        """
        # 默认最大上下文长度为训练时的最大长度
        if max_context_len == -1:
            max_context_len = self.max_seq_len
            
        present_key_value = None  # kv缓存
        for _ in range(max_new_tokens):
            # 是否使用kv缓存, 并且裁剪长度
            if use_cache:
                if present_key_value is None:
                    # 第一次推理，此时kv还没缓存，需要输入整个序列
                    input_ids = ids
                else:
                    # 后续推理，此时kv已经缓存，只需要输入最后一个位置的token
                    input_ids = ids[:, [-1]]
            
                # 裁剪kv缓存中的长度 <= max_context_len - 1, 注意query还占一个位置
                # self.max_seq_len代表训练时的最大长度，通过RoPE的长度外推，实际推理时可以使用更长的序列
                # (bsz, n_kv_heads, seq_len, head_dim)
                if present_key_value is not None and present_key_value['seen_tokens'] >= max_context_len:
                    present_key_value['key_states_layers'] = [k[:, :, 1-max_context_len:, :] for k in present_key_value['key_states_layers']]
                    present_key_value['value_states_layers'] = [v[:, :, 1-max_context_len:, :] for v in present_key_value['value_states_layers']]
                    present_key_value['seen_tokens'] = max_context_len - 1
            else:
                # 判断是否裁剪长度
                input_ids = ids if ids.size(-1) <= max_context_len else ids[:, -max_context_len:]
            
            # 前向传播模型，得到序列中索引的logits
            model_outputs = self(input_ids, use_cache=use_cache, past_key_values=present_key_value)
            logits = model_outputs.logits[:, -1, :] # 只取最后一个位置的输出, (bsz, vocab_size)
            
            # 采样
            if temperature == 0.0:
                # 采样单个最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 从概率分布中采样一个索引
                # 将最后一步的logits取出并按所需的温度进行缩放
                logits = logits / temperature
                # 可选，将logits裁剪到前k个选项
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')  # 将小于第k大的logits设置为负无穷
                # 应用softmax将logits转换为（归一化的）概率
                probs = F.softmax(logits, dim=-1)
                # 从概率分布中采样一个索引
                idx_next = torch.multinomial(probs, num_samples=1)

            # 将采样的索引添加到序列中并继续推理
            ids = torch.cat((ids, idx_next), dim=1)

        return ids
