"""
llama2模型结构

1）注意在RMS、RoPE、attention中的softmax计算时，使用的是float32，并且计算完要转换回原来的类型（如bfloat16）
https://zhuanlan.zhihu.com/p/651588659

2）无论使用 BF16 还是 FP16，都有一个权重副本始终在 FP32 中，这是由优化器更新的内容。
16 位格式仅用于计算，优化器以全精度更新 FP32 权重，然后将它们转换为 16 位格式以用于下一次迭代。因此，不会发生精度损失。
https://www.zhihu.com/question/616600181/answer/3160547952?utm_source=zhihu

"""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_naive import Z2allConfig


ACT2FN = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "silu": nn.SiLU(),
    "swish": nn.SiLU(),
    "tanh": nn.Tanh(),
}


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 计算时用float32
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
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
            self.multiple_of = config.multiple_of  # 保持intermediate_size是multiple_of的倍数
            self.intermediate_size = self.multiple_of * ((self.intermediate_size + self.multiple_of - 1) // self.multiple_of)
        
        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            # 预训练时的张量并行度, tensor parallelism
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效操作。
    隐藏状态从 (batch, n_kv_heads, seq_len, head_dim) 变为 (batch, n_heads, seq_len, head_dim)
    """
    batch, n_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


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
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, self.max_seq_len, self.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # model.to("cuda")时，在buffer中的tensor同样会被移动到cuda
            self.register_buffer("mask", mask)

    @staticmethod
    def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin, position_ids):
        # 计算时用float32
        # q, k 可能是 bfloat16
        # freqs_cos, freqs_sin 是 float32
        
        # q, k: (bsz, n_heads, q_len, head_dim)
        # freqs_cos -> (1, 1, q_len, head_dim) [..., [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]]
        freqs_cos = freqs_cos[None, None, position_ids, :]
        freqs_sin = freqs_sin[None, None, position_ids, :]

        # rotate_half_q (bsz, n_heads, q_len, head_dim) [..., [-q1, q0, -q3, q2, ...]]
        rotate_half_q = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q)
        rotate_half_k = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(k)

        q_embed = (q * freqs_cos) + (rotate_half_q * freqs_sin)
        k_embed = (k * freqs_cos) + (rotate_half_k * freqs_sin)
        return q_embed.type_as(q), k_embed.type_as(k)

    def update_key_value_cache(self, past_key_value, key_states, value_states, layer_idx):
        # key_states, value_states: (bsz, n_kv_heads, seq_len, head_dim)

        # 更新 seen_tokens
        if layer_idx == 0:
            past_key_value['seen_tokens'] += key_states.shape[-2]

        # 更新缓存值
        if len(past_key_value['key_states_layers']) <= layer_idx:
            past_key_value['key_states_layers'].append(key_states)
            past_key_value['value_states_layers'].append(value_states)
        else:
            past_key_value['key_states_layers'][layer_idx] = torch.cat([past_key_value['key_states_layers'][layer_idx], key_states], dim=-2)
            past_key_value['value_states_layers'][layer_idx] = torch.cat([past_key_value['value_states_layers'][layer_idx], value_states], dim=-2)

        return past_key_value

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效操作。
        """
        # hidden_states (bsz, n_kv_heads, seq_len, head_dim)
        bsz, n_kv_heads, seq_len, head_dim = hidden_states.shape

        if n_rep == 1:
            return hidden_states

        # hidden_states (bsz, n_kv_heads, seq_len, head_dim) -> (bsz, n_kv_heads * n_rep, seq_len, head_dim)
        hidden_states = (
            hidden_states[:, :, None, :, :]
            .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
            .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
        )

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # hidden_states: (bsz, q_len, hidden_dim)
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            # 预训练时的张量并行度，对结果不影响, tensor parallelism
            # Linear，在输出维度上分割，则结果用cat
            # Linear，在输入维度上分割，则结果用sum
            query_slicing = (self.n_heads * self.head_dim) // self.config.pretraining_tp
            key_value_slicing = (self.n_kv_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(query_slicing, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # (bsz, q_len, hidden_dim) -> (bsz, n_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, freqs_cos, freqs_sin, position_ids)

        # 使用kv缓存
        if use_cache and past_key_value is not None:
            past_key_value = self.update_key_value_cache(past_key_value, key_states, value_states, self.layer_idx)
            key_states = past_key_value['key_states_layers'][self.layer_idx]
            value_states = past_key_value['value_states_layers'][self.layer_idx]

        # GQA，分组查询注意力
        # key_states (bsz, n_kv_heads, all_seq_len, head_dim) -> (bsz, n_heads, all_seq_len, head_dim)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # flash implementation
        if self.flash:
            attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # query_states (bsz, n_heads, q_len, head_dim)
            # key_states (bsz, n_heads, all_seq_len, head_dim)
            # key_states包含之前缓存的长度
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            assert hasattr(self, 'mask')
            attn_weights = attn_weights + self.mask[:, :, position_ids, :]   # (bsz, n_heads, q_len, all_seq_len)

            # 注意力计算时用fp32
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            # attn_weights (bsz, n_heads, q_len, all_seq_len)
            # value_states (bsz, n_heads, all_seq_len, head_dim)
            attn_output = torch.matmul(attn_weights, value_states)

        # attn_output (bsz, n_heads, q_len, head_dim) -> (bsz, q_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_dim)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_dim // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_dim // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class Z2allDecoderLayer(nn.Module):
    def __init__(self, config: Z2allConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        self.self_attn = Z2allAttention(config=config, layer_idx=layer_idx)

        self.mlp = Z2allMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


def init_key_value_cache():
    return {"key_states_layers": [], "value_states_layers": [], "seen_tokens": 0}


class Z2allPreTrainedModel(nn.Module):
    """
    用于权重的初始化
    """
    
    def _init_weights(self, module):
        std = self.config.initializer_range  # 默认为0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def post_init(self):
        # 初始化权重，应用到模型的所有子模块
        self.apply(self._init_weights)
        # 应用特殊的缩放初始化到残差投影，参考GPT-2论文，（见llama2.c项目）
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.n_layers))


class Z2allModel(Z2allPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = config.hidden_dim
        self.n_heads = config.n_heads
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.n_layers = config.n_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim, self.padding_idx)
        self.layers = nn.ModuleList(
            [Z2allDecoderLayer(config, layer_idx) for layer_idx in range(self.n_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_norm_eps)
        self.output = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 是否共享参数
        if config.tie_word_embeddings:
            self.embed_tokens.weight = self.output.weight
        
        # 预处理RoPE中用到的cos和sin
        self.freqs_cos, self.freqs_sin = self.precompute_cos_sin(
            dim=self.hidden_dim // self.n_heads,
            end=self.max_seq_len * 2,  # 注意，end = self.max_seq_len * 2，因为Llama 2模型的token限制为4096。
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
        )
        # 弃用，防止其因混合精度转换为bfloat16，导致精度损失
        # # model.to("cuda")时，在buffer中的tensor同样会被移动到cuda
        # self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        # self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # 初始化权重
        self.post_init()

    def rotary_embedding(self, dim, position_ids, base=10000, device=None):
        seq_len = torch.max(position_ids) + 1

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

        position_ids_expanded = position_ids[:, None].float().to(device)  # [seq_len, 1]
        inv_freq_expanded = inv_freq[None, :].float().to(device)  # [1, dim//2]
        freqs = (position_ids_expanded @ inv_freq_expanded)  # [seq_len, dim//2]
        # [θ0,θ1,θ2......θd/2-1] -> [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        emb = torch.stack((freqs, freqs), dim=-1).reshape(seq_len, dim)  # [seq_len, dim]

        cos, sin = emb.cos(), emb.sin()

        return cos, sin

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
        labels: Optional[torch.Tensor] = None,
        use_cache=False,
        past_key_value: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """模型的前向传播
        kv缓存的格式: {"key_states_layers": [], "value_states_layers": [], "seen_tokens": 0}

        Args:
            input_ids (torch.Tensor): 输入的token id
            labels (torch.Tensor, optional): 预训练时的标签. Defaults to None.
            use_cache (bool): 是否使用kv缓存. Defaults to False.
            past_key_value (_type_, optional): kv缓存. Defaults to None.

        Returns:
            Tuple: (输出的logits, kv缓存)
        """
        _, seq_len = input_ids.size()
        
        start_pos = 0
        # 初始化kv缓存
        if use_cache:
            if past_key_value is None:
                past_key_value = init_key_value_cache()
            else:
                start_pos = past_key_value['seen_tokens']
        
        all_seq_len = start_pos + seq_len  # 包含了之前缓存的token长度
        
        # 得到position_ids
        position_ids = torch.arange(start_pos, all_seq_len, device=input_ids.device, dtype=torch.int64)
        
        # 将cos, sin放入cuda, float32
        self.freqs_cos = self.freqs_cos.to(input_ids.device)
        self.freqs_sin = self.freqs_sin.to(input_ids.device)
            
        # token嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        present_key_value = None
        for decoder_layer in self.layers:
            hidden_states, present_key_value = decoder_layer(
                hidden_states=hidden_states,
                freqs_cos=self.freqs_cos,
                freqs_sin=self.freqs_sin,
                position_ids=position_ids,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
        hidden_states = self.norm(hidden_states)
        
        if labels is not None:
            # 给定标签，计算损失
            logits = self.output(hidden_states)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # 推理时，只需要最后一个位置的输出
            # logits (bsz, 1, vocab_size)
            logits = self.output(hidden_states[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
        
        return logits, present_key_value

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
            logits, present_key_value = self(input_ids, use_cache=use_cache, past_key_value=present_key_value)
            logits = logits[:, -1, :] # 只取最后一个位置的输出, (bsz, vocab_size)
            
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
