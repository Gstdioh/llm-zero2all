import torch

# 修改路径：
# miniconda3/envs/LLM/lib/python3.10/site-packages/torch/cuda/amp/autocast_mode.py
# 修改方法见：
# https://github.com/pytorch/pytorch/commit/bc03aa6013e101222c9652d04a2b08e48f626dfb#diff-dac4bd53ced015c8810b3b02fc5c2ec6c2b0658c5090b4fbbd09c96bd45087d1

class SomeFunc(torch.autograd.Function):
    @classmethod
    @torch.cuda.amp.custom_fwd
    def forward(cls, ctx, x, y):
        out = x @ y
        print("Forward dtype:", out.dtype)
        ctx.save_for_backward(x, y)
        return out

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, grad):
        x, y = ctx.saved_tensors
        grad_x = grad @ y.T
        grad_y = x.T @ grad
        print("Backward dtype:", grad_x.dtype, grad_y.dtype)
        print(grad_x)
        print(grad_y)
        return grad_x, grad_y

x = torch.randn([2, 3], device="cuda", requires_grad=True)
y = torch.randn([3, 4], device="cuda", requires_grad=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    z = SomeFunc.apply(x, y)
    z.backward(torch.ones_like(z))

print(x.grad)
print(y.grad)

print(1)
