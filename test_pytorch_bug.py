import torch

# 修改路径：
# miniconda3/envs/LLM/lib/python3.10/site-packages/torch/cuda/amp/autocast_mode.py
# 修改方法见：
# https://github.com/pytorch/pytorch/commit/bc03aa6013e101222c9652d04a2b08e48f626dfb#diff-dac4bd53ced015c8810b3b02fc5c2ec6c2b0658c5090b4fbbd09c96bd45087d1

class SomeFunc(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd
    def forward(cls, ctx, x):
        out = x.transpose(0, 1) @ x
        print("Forward dtype:", out.dtype)
        ctx.save_for_backward(out, x)
        return out

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad):
        out, x = ctx.saved_tensors
        out_recalc = x.transpose(0, 1) @ x
        print("Backward dtype:", out_recalc.dtype)
        return None

x = torch.randn([10, 10], device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = SomeFunc.apply(x)
y.backward(torch.zeros_like(y))