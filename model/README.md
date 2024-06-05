# 模型结构 Z2all (使用 llama2 结构)

## 将数值大的放在第一维度，这样可以提高GPU的并行度?

hidden_states (bsz, seq_len, hidden_dim) -> (seq_len, bsz, hidden_dim)

见：https://arxiv.org/pdf/2104.04473.pdf，也有人问了这个问题：https://github.com/NVIDIA/Megatron-LM/issues/493，但是没得到确切的回答

这是Megatron-LLM的做法，在这里我还是有疑问，虽然说将s放在第一维对seq并行有好处，但是将s放在第一维是Megatron在第二篇论文提到的，seq并行却是第三篇论文提到的。。。所以还是不清楚为什么要将数值大的放在第一维度

## 融合算子
包括flash-attn, rope, cross_entropy, rmsnorm, swiglu, AdamW
训练设置：1个A800-80G，每个迭代训练的token数：

tokens per iteration will be: 262,144

breaks down as: 16 grad accum steps * 1 processes * 2 batch size * 2048 max seq len

训练命令：`python pretrain.py --batch_size=2 --gradient_accumulation_steps=16`

模型超参数：

```python
max_seq_len = 2048
batch_size = 2
gradient_accumulation_steps = 16

vocab_size = 64320  # 实际是64012个，写大点方便扩展，注意最好是8的倍数，见指导：https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#tc-guidelines-padding
hidden_dim = 2048
intermediate_size = 5632
n_layers = 22
n_heads = 32
n_kv_heads = 8  # 用于GQA
max_seq_len = max_seq_len
initializer_range = 0.02  # 参数初始化时的标准差
rms_norm_eps = 1e-5  # 防止除0的小数
pad_token_id = 64006  # tokenizer.special_tokens["<|PAD|>"]  # pad token 64006
tie_word_embeddings = False  # 是否共享word embedding和word prediction的参数
rope_theta = 10000.0
rope_scaling = None  # 缩放方法，用于长度外推
attention_bias = False  # attention中的project是否加bias，Qwen中加了
attention_dropout = 0.0  # TODO: 或许不用设置dropout
dropout1 = 0.0
dropout2 = 0.0
residual_in_fp32 = True  # 残差连接是否使用fp32
```

模型和优化器状态占用内存：12250MB = 12.25GB

---

pytorch1.12.1和cuda11.4下，第5个iter（从0开始）的性能比较：

| 融合算子           | 速度比较/s | MFU (Model FLOPs Utilization)/% | 内存占用/GB |
| :---------------: | :-------: | :----------------------------: | :---------: |
| navie             | 9.85      | 18.41                          | 71.32       |
| **use_all**       | **3.23**  | **56.06**                      | **27.87**   |
| w/o flash-attn    | 8.69      | 20.87                          | 65.43       |
| w/o rope          | 3.51      | 51.70                          | 27.90       |
| w/o cross_entropy | 3.37      | 53.70                          | 29.83       |
| w/o rmsnorm       | 3.78      | 47.94                          | 30.68       |
| w/o swiglu        | 3.60      | 50.39                          | 29.79       |

在上面use_all基础上再加入额外的优化（主要不想再跑一遍上面的消融了haha。。。）：

| 融合算子           | 速度比较/s | MFU (Model FLOPs Utilization)/% | 内存占用/GB |
| :---------------: | :-------: | :----------------------------: | :---------: |
| **w apex_AdamW**  | **3.17**  | **57.47**                      | **27.89**   |

---

pytorch2.0.1和cuda11.4下，不使用compile，第5个iter（从0开始）的性能比较：

| 融合算子           | 速度比较/s | MFU (Model FLOPs Utilization)/% | 内存占用/GB |
| :---------------: | :-------: | :----------------------------: | :---------: |
| navie             | 9.72      | 18.66                          | 72.85       |
| **use_all**       | **3.49**  | **51.95**                      | **27.95**   |
| flash-attn->sdpa  | 3.85      | 47.09                          | 28.89       |
| w/o flash-attn    | 8.75      | 20.72                          | 65.47       |
| w/o rope          | 3.68      | 49.18                          | 27.96       |
| w/o cross_entropy | 3.55      | 51.02                          | 29.89       |
| w/o rmsnorm       | 4.00      | 45.27                          | 32.15       |
| w/o swiglu        | 3.60      | 50.38                          | 29.86       |
| w/o AdamW         | 3.52      | 51.55                          | 30.52       |

---

pytorch2.0.1和cuda11.4下，**使用compile**（与fused_swiglu不兼容，compile的作用可类似于融合算子，所以加上某些自定义的融合算子反而会降低性能，带上compile后性能有浮动，比较的结果看看就行），第5个iter（从0开始）的性能比较：

| 融合算子           | 速度比较/s | MFU (Model FLOPs Utilization)/% | 内存占用/GB |
| :---------------: | :-------: | :----------------------------: | :---------: |
| navie             | 5.47      | 33.14                          | 64.97       |
| use_all           | 3.60      | 49.23                          | 32.46       |
| flash-attn->sdpa  | 3.95      | 45.91                          | 33.46       |
| w/o flash-attn    | 5.60      | 32.37                          | 65.92       |
| w/o rope          | 3.51      | 51.67                          | 31.58       |
| w/o cross_entropy | 3.64      | 49.74                          | 33.65       |
| w/o rmsnorm       | 3.45      | 52.46                          | 32.41       |
| w/o swiglu        | error     | error                          | error       |
| w/o AdamW         | 3.57      | 50.69                          | 34.05       |
| just flash-attn   | 3.44      | 52.73                          | 32.76       |
|**flash,cross_entropy,AdamW**| **3.37** | **53.71**                   | **31.54**   |

---

最后我选择pytorch1.12.1_cuda11.4的版本进行训练。

---

以下是各种所需包的构建过程

通常会需要ninja和packagin这两个包来编译，ninja用来加速编译
```bash
# 加速安装
pip install ninja
pip install packaging
```

### 1 flash-attn
flash-attn的实现比torch_flash的实现更快

flash attention可用于训练和推理，训练减少显存占用的原理是重计算，虽然重计算会增加FLOPs，但是不需要从HBM中取中间结果了，即减少了IO，所以依然减少了运行时间。

```bash
# 2.1.0需要pytorch>=1.12, cuda>=11.4
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout tags/v2.1.0

# 不要用`pip install .`安装
python setup.py install
```

### 2 rotary_emb
```bash
# 继续在flash-attention中
cd flash-attention/csrc/rotary
pip install .
```

### 3 xentropy_cuda_lib
```bash
# 继续在flash-attention中
cd flash-attention/csrc/xentropy
pip install .
```

### 4 dropout_layer_norm
flash-attention库中的，在cuda11.4下安装会卡住，或许可以尝试xformers库的rms_norm_add这个来替代（但是rms_norm_add好像与cuda11.4也不兼容。。。）

```bash
# 继续在flash-attention中，注意在cuda11.4下安装会卡住，同时pytorch2.1.0下安装有问题，版本不兼容
# 我直接用的qwenllm/qwen latest镜像，已经安装好了，环境是pytorch2.0.1和cuda11.7
# 但是我的训练环境因为只有cuda11.4驱动，也不好更新驱动，所以实际训练时没有用这个包
cd flash-attention/csrc/layer_norm
pip install .
```

### 5 MixedFusedRMSNorm (apex)
apex中有MixedFusedRMSNorm，如果安装不了上面的dropout_layer_norm，可以用这个来加速

在apex的最新版本中，多了一个memory_efficient参数，这个开关可以在速度和精度无损的情况下节省网络训练的显存占用，见：https://zhuanlan.zhihu.com/p/677986216，但是安装最新版本有问题，我就没有使用。

在 cuda11.4, pytorch1.12.1_cuda11.3 环境下安装时，需要注释掉apex的setup.py中的 `check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)`

**注意**，pytorch1.12.1下安装apex 22.04-dev，pytorch2.0.1下安装apex tags/23.05

安装见：https://zhuanlan.zhihu.com/p/672284687
```bash
# 获取apex, 注意这里一定要通过git clone 不要自己下载zip包，不然就会碰到错误4
git clone https://github.com/NVIDIA/apex.git

cd apex
git branch -a
# git checkout -b 22.04-dev  origin/22.04-dev #切换分支，当前的主分支有问题，你会碰到错误5
git checkout tags/23.05  # 22.04-dev和pytorch2.0.1又有冲突了。。。，这个23.05可以用

pip uninstall apex  #卸载之前的apex

# 编译apex，要加 --disable-pip-version-check 不然你会碰到错误2
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./

等待10分钟....
```

### 6 SwiGLU (xformers)
这个算子在xformers库中实现，注意需要ninja来加速编译

```bash
# v0.0.23时，torch >= 1.12
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout tags/v0.0.23

# 导入子模块中的第三方仓库
git submodule update --init --recursive

python setup.py install
```

**注意**，在xformers中，如果要使用autocast的自动混合精度，那么要删除xformers/ops/swiglu_op.py中`_ForwardToPythonAutogradFunc`中的这一段代码，同时需要解决一个pytorch<=1.12.1的一个bug（目前我测试pytorch2.0.1中修复了这个bug）：

```python
if op.dtype_autocast_gpu == torch.bfloat16:
    return False
```

在pytorch2.0.0下已经修复了这个bug，我在xformers仓库提交了一个pull request，将上述代码改为了：

```python
if op.dtype_autocast_gpu == torch.bfloat16 and torch.__version__ < "2.0.0":
    return False
```

**注意**，pytorch<=1.12.1有一个bug，见`./test_pytorch_bug.py`文件，运行该文件，你会发现反向传播时将bfloat16错误转换为float16

问题：使用autocast时，反向传播会将bfloat16错误转换为float16，解决方法：

1. 升级pytorch
2. 根据[链接](https://github.com/pytorch/pytorch/commit/bc03aa6013e101222c9652d04a2b08e48f626dfb#diff-dac4bd53ced015c8810b3b02fc5c2ec6c2b0658c5090b4fbbd09c96bd45087d1)
来修改你的pytorch源码。
3. 我提供了修改完的代码，将`new_autocast_mode.py`的内容复制到你的pytorch的`autocast.mode.py`中即可。（推荐）

注意，通过ctrl+左键点击`@torch.cuda.amp.custom_fwd`的custom_fwd即可跳转到你的pytorch的`autocast.mode.py`中。

**总结**，注释xformers中的代码，修复pytorch<2.0.0时的bug。

### 7 fused AdamW
两种方法，pytorch>=2.0.0使用pytorch官方的fuesd AdamW，pytorch<2.0.0使用apex0.1的FusedAdam

1. 需要pytorch>=2.0.0，使用代码见：[`./utils/train.py`](./utils/train.py)

2. 安装apex，调用`from apex.optimizers import FusedAdam`，注意optim.zero_grad()时和上者有区别，如下：

```python
if utils.FusedAdam is not None:
    optimizer.zero_grad()  # apex fused adamw上已经设置了set_to_none了
else:
    optimizer.zero_grad(set_to_none=True)  # pytorch的需要在这设置为None，清空显存
```
