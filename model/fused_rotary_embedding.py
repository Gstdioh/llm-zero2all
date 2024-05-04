# Copyright (c) 2023, Tri Dao.

import math
from typing import Optional, Tuple

import rotary_emb
import torch
from einops import rearrange, repeat

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        """
        源码：lit_gpt/fused_rotary_embedding.py
        注意，我这里修改了x的维度，x的维度为(seqlen, batch_size, nheads, headdim)
            cos, sin的维度要和x的维度一致即(seqlen, 1, 1, headdim)
            
            x: (batch_size, seqlen, nheads, headdim) -> (seqlen, batch_size, nheads, headdim)
            cos, sin: (seqlen, headdim / 2) -> ((seqlen, 1, 1, headdim / 2))
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
                of 1st half and 2nd half (GPT-NeoX style).
        rotary_dim must be <= headdim
        Apply rotary embedding to the first rotary_dim of x.
        """
        headdim = x.shape[-1]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        x_ro = x[..., :rotary_dim]
        x1, x2 = x_ro.chunk(2, dim=-1) if not interleaved else (x_ro[..., ::2], x_ro[..., 1::2])
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            x1,
            x2,
            cos,  # (seq_len, 1, 1, head_dim / 2)  通常来说，rotary_dim = headdim
            sin,
            # rearrange(cos[:seqlen], "s d -> s 1 1 d"),
            # rearrange(sin[:seqlen], "s d -> s 1 1 d"),
            o1,
            o2,
            False,
        )
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        headdim = do.shape[-1]
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1) if not ctx.interleaved else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            cos,  # (seq_len, 1, 1, head_dim / 2)
            sin,
            # rearrange(cos[:seqlen], "s d -> s 1 1 d"),
            # rearrange(sin[:seqlen], "s d -> s 1 1 d"),
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


def fused_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    return ApplyRotaryEmb.apply(x, cos, sin, interleaved, inplace)

# apply_rotary_emb_func = ApplyRotaryEmb.apply
