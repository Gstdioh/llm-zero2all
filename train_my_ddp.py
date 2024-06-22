"""
### train_my_ddp.py 训练脚本

1. 支持pretrain, sft, dpo任务
2. 支持单卡，单节点多卡，多节点多卡
3. 支持resume，需要添加 --resume --out_dir=out/2024_06_06_22_23_57

会自动搜索对应目录中的相应文件

多卡下会有警告，可以添加 OMP_NUM_THREADS=8 环境变量解决

运行快点用于测试，添加参数（注意gradient_accumulation_steps要能被ddp_world_size整除）：--train_batch_size=2 --gradient_accumulation_steps=3

#### pretrain

```bash
# 1. pretokenize, [".parquet", ".json", ".jsonl"] -> [".bin"]
python -m data_preprocess.pretokenize_pretrain_data --data_dir=data/02_train_data_more --tokenizer_dir=tokenizer/hf_bbpe_tokenizer

# 2. 预处理，构建索引, [".bin"] -> [".ibin"]
python -m data_preprocess.build_pretrain_sample_index_map --data_dir=data/02_train_data_more --max_seq_len=2048

# 3. train
# 单卡
python train_my_ddp.py --task_type=pretrain --data_dir=data/02_train_data_more

# 单节点多卡
torchrun --standalone --nproc_per_node=4 train_my_ddp.py --task_type=pretrain --data_dir=data/02_train_data_more

# 多节点多卡
# - master
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 train_my_ddp.py --task_type=pretrain --data_dir=data/02_train_data_more
# - remote
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 train_my_ddp.py --task_type=pretrain --data_dir=data/02_train_data_more
```

#### sft

```bash
1. pretokenize，且构建索引（可不进行这一步，在线构建也可以，然后会自动保存为cache）
python -m data_preprocess.pretokenize_sft_data --data_dir=data/04_sft_conversation_data --tokenizer_dir=tokenizer/hf_bbpe_tokenizer

2. train
# 单卡

# - 使用pretokenize后的bin文件，只支持 sft_type=["conversation"]
python train_my_ddp.py --task_type=sft --data_dir=data/04_sft_conversation_data/01_bin_for_sft --sft_type=conversation

# - 使用原始的json，支持 sft_type=["conversation", "instruction"]
python train_my_ddp.py --task_type=sft --data_dir=data/03_sft_data --sft_type=instruction --use_dataset_with_index=False
python train_my_ddp.py --task_type=sft --data_dir=data/04_sft_conversation_data/train_3.5M_CN --sft_type=conversation --use_dataset_with_index=False
```

#### dpo

```bash
# 1. train，不支持pretokenize，会进行在线构建，然后会自动保存为cache，只支持conversation格式
python train_my_ddp.py --task_type=dpo --data_dir=data/05_dpo_data/DPO-En-Zh-20k
```
"""

import math
import os
import inspect
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
import logging

import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed
from transformers import AutoConfig, AutoTokenizer

from model import Z2allConfig, Z2allForCausalLM
from utils import my_logging, ResLog
from utils import print_rank0, save_run_exp_config
from utils.checkpoint import save_checkpoint, copy_tensor_to_device_in_object
from utils.train import configure_optimizers, estimate_mfu, forward_step
from utils.dpo import DPOConfig, create_reference_model

from parallel.distributed_data_parallel import DistributedDataParallelConfig
from parallel.distributed_data_parallel import DistributedDataParallel as MyDDP
from optimizer import OptimizerConfig, FP32Optimizer, Float16OptimizerWithFloat16Params
from parallel.distributed_optimizer import DistributedOptimizer
from parallel.distributed_data_parallel.ddp_comm_hooks.default_hooks import all_reduce_hook, reduce_scatter_hook, bf16_compress_wrapper, stream_wrapper
from parallel.distributed_data_parallel.ddp_comm_hooks.overlap_optim_step_hooks import overlap_optim_step_wrapper
from parallel.distributed_data_parallel.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

from train_task import Task


# 前两个多节点需要，第三个多卡需要
os.environ["NCCL_IB_DISABLE"] = "1"  # disable infiniband
os.environ["NCCL_IBEXT_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"  # disable p2p
# os.environ["OMP_NUM_THREADS"] = "8"  # set the number of threads for OpenMP，这样设置好像没用，因为是torchrun用到的，还是需要运行时设置OMP_NUM_THREADS=8
# os.environ["NCCL_DEBUG"] = "WARN"  # set NCCL debug level, ["WARN", "INFO"]，用于测试

# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # 设置每个 CUDA 设备的最大并行内核执行数，速度还快了？

# 用于找到pad的id
# tokenizer = AutoTokenizer.from_pretrained("tokenizer/hf_bbpe_tokenizer", trust_remote_code=True)
# pad_token_id = tokenizer.pad_token_id

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
iter_num = 0  # 从该iter开始训练
log_interval = 1  # 每log_interval个step打印一次信息
resume = False  # if True, resume training from the last checkpoint
resume_from = "last"  # ["best", "last"] 从哪个开始resume
sync_for_true_time = False  # 是否同步以获取mirco的真实耗费的时间，测试时用
# -----------------------------------------------------------------------------
# my logging
use_reslog = True  # wandb用起来有问题，改为自己的日志和画图工具，这个必须为True，因为还会被用来判断文件是否保存成功
reslog_dir = "reslog"
reslog_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
reslog_save_interval = 10  # 想快速看结果，可以用小点的数
# -----------------------------------------------------------------------------
# data，要么使用data_dir，要么使用（train_data_dir, valid_data_dir）
data_dir = "data/02_train_data_more"
valid_ratio = 1e-4  # 从data_dir中自动划分出来的验证集的比例，pretrain下可以设小点，如共有13_000个iters，1e-4即约1个iters，sft和dpo可以设置为1e-2
# 若分别提供train_data_dir和valid_data_dir，则不会从data_dir中划分验证集
train_data_dir = None
valid_data_dir = None
# train
train_batch_size = 8  # 每个step的train_batch_size
gradient_accumulation_steps = 128  # used to simulate larger batch sizes，后面会//ddp_world_size
# valid
valid_interval = 200  # 每valid_interval个step验证一次
valid_batch_size = 8  # 用于验证的batch_size
valid_iters = None  # 每次验证的step数，若没有提供，则验证完整的数据集，train也会验证相同数量的样本
valid_only = False  # 只进行验证一次就退出
# 其他
max_seq_len = 2048  # 每个样本的最大长度
grad_div_total_tokens = False  # 是否在计算梯度时除以总的token数，设置reduction="none" and grad_scaling_before_comm=False，使用PowerSGD时不能使用（loss不好，可能因为PowerSGD对数大的压缩不好，有正交化操作）
num_workers = 0  # 数据加载器的工作进程数
skip_scaling_factor = 1.0  # 跳过的数据集数可能需要乘的数，因为非index的pretrain_dataset构造方式有点不同，GPU个数变化的时候需要设置，正常设置1.0即可
# -----------------------------------------------------------------------------
# dataset的设置
# dataset, task
task_type = "pretrain"  # pretrain|sft|dpo
# pretrain, sft
use_dataset_with_index = True  # pretrain和sft下可用，是否使用索引来遍历数据集，需要先通过build_sample_index_map.py构建sample的索引，建议使用True！pretrain下False的话后续更改gpu数量再训练会很麻烦（需要调整跳过的iter数）
# sft, dpo
tokenizer_dir = "tokenizer/hf_bbpe_tokenizer"
user_name = "human"
assistant_name = "gpt"
# sft
sft_type = "conversation"  # sft任务的类型，"conversation"（多轮），"instruction"（单轮指令）
# dpo
max_prompt_len = None  # 默认为max_seq_len//2
prompt_truncation_mode = "keep_end"  # prompt保留的方式，"keep_end"（保留后面），"keep_start"（保留前面）
num_cpus = 0  # dpo预处理时使用的进程数，并行处理的进程数（>= 2则会启动多进程），太大可能会卡住，设置为2还是会报错（不清楚为什么。。。）, RuntimeError: unable to mmap 704 bytes from file <filename not specified>: Cannot allocate memory (12)
# -----------------------------------------------------------------------------
# peft
use_peft = False
# dpo_config
ignore_index = -100
dpo_loss_type = "sigmoid"
dpo_beta = 0.01
dpo_label_smoothing = 0.0
reference_free = False
# -----------------------------------------------------------------------------
# model
vocab_size = 64320  # 实际是64012个，写大点方便扩展，注意最好是8的倍数，见指导：https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#tc-guidelines-padding
hidden_dim = 2048
intermediate_size = 5632
n_layers = 32
n_heads = 16
n_kv_heads = 8  # 用于GQA
max_seq_len = max_seq_len
initializer_range = 0.02  # 参数初始化时的标准差
rms_norm_eps = 1e-5  # 防止除0的小数
pad_token_id = 64006  # tokenizer.pad_token_id  # pad token 64006
tie_word_embeddings = False  # 是否共享word embedding和word prediction的参数
rope_theta = 10000.0
rope_scaling = None  # 缩放方法，用于长度外推
attention_bias = True  # attention中的project是否加bias，Qwen中加了
attention_dropout = 0.05  # TODO: 或许不用设置dropout
dropout1 = 0.05
dropout2 = 0.05
residual_in_fp32 = True  # 残差连接是否使用fp32
loss_reduction = "none" if grad_div_total_tokens else "mean"  # 损失函数的reduction方式，"mean" or "none"，使用"none"可以和grad_scaling_before_comm=False配合使用，减少精度损失
# -----------------------------------------------------------------------------
# adamw optimizer
learning_rate = 3e-4  # max learning rate，参考Qwen
epoch = 2.2  # 我训练共有26B个token，token batch大小为2M，即一个epoch共有13_000个iters，max_iters=epoch*13_000，max_seq_len为2048，即一个epoch共有13M个samples
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# -----------------------------------------------------------------------------
# learning rate decay settings
decay_lr = True  # 是否进行学习率的衰减
warmup_ratio = 0.03  # warmup所占max_iters的比例
# -----------------------------------------------------------------------------
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# 分布式配置
# 通信后端
# 见：https://huggingface.co/docs/transformers/perf_train_gpu_many
# 使用nccl时，如果没有nvlink，则需要设置NCCL_P2P_DISABLE=1
# 没有nvlink时，在单节点下DP比DDP更快，但是DP不支持多节点训练
# 因为我的环境没有nvlink，所以我使用的是gloo后端
# 但是gloo又与hook有问题，还是用nccl吧
# 1. gloo，不支持bfloat16，使用PowerSGD时会卡住，强行退出时GPU不会立即释放
# 2. nccl，需要设置NCCL_IB_DISABLE=1，NCCL_IBEXT_DISABLE=1，NCCL_P2P_DISABLE=1，前两个多节点需要，第三个多卡需要
ddp_backend = "nccl"  # ddp backend, can be 'nccl', 'gloo'
# 梯度通信优化
use_bf16_compress_hook = False
# powerSGD相关参数
use_powerSGD_hook = False
matrix_approximation_rank = 32  # 用于PowerSGD的矩阵近似秩，如矩阵m * n -> m * rank, rank * n
warm_start = False  # Q是否沿用上一次iter的值
use_error_feedback = True  # 是否使用error feedback，即将error传递给下一次iter
start_powerSGD_iter = 2  # 从第几个iter开始使用PowerSGD，至少为2
min_compression_rate = 2  # 能压缩多少才进行压缩，如为2，则表示能压缩到原来的1/2的矩阵才压缩，不能则为uncompress_tensor
orthogonalization_epsilon = 1e-6  # 正交时的epsilon，防止除以0，float16和bfloat16下会用到
grad_buffer_is_powerSGD_error = True and use_powerSGD_hook  # 将grad_buffer和error_dict的内存空间共享，可以节省模型梯度大小的内存
orthogonalize_in_float32 = True  # 设置正交化操作在float32上进行，可以提高精度
use_fixed_Q = True  # 是否使用固定的Q矩阵，不再更新Q矩阵，参考：DALL-E: Zero-Shot Text-to-Image Generation
# DistributedDataParallelConfig
grad_reduce_in_fp32 = False  # 梯度的buffer设置为fp32，也表示累加的精度
overlap_grad_reduce = True  # 梯度通信与backward计算重叠
use_distributed_optimizer = False  # 是否使用DistributedOptimizer
check_for_nan_in_grad = False  # 在bucket进行梯度通信时，检查梯度是否有nan，有则报错
bucket_size = 10_000_000  # 一个bucket的最大大小，超过则分割
disable_bucketing = False  # 是否禁用bucket，即最后进行整个模型的梯度通信
# OptimizerConfig
precision_dtype = dtype  # 使用的精度，若不为float32，则表示开启混合精度
grad_scaling_before_comm = False if grad_div_total_tokens else True  # 是否在通信前进行梯度缩放，建议bfloat16下设为False，在最后除以值，减少精度损失
overlap_optim_step = True  # 某个bucket的梯度算完后（通信后），立刻进行优化器的step，有梯度通信的情况下会计算与通信重叠
overlap_zero_grad_buffer = True  # overlap_optim_step后立刻对模型的grad_buffer清零，注意在powerSGD的grad_buffer_is_powerSGD_error下不会清零，而是计算为error
grad_buffer_is_powerSGD_error = grad_buffer_is_powerSGD_error # 梯度缓冲区是否是PowerSGD的error缓冲区，如果是，则不需要清零，这样可以节省内存
use_distributed_optimizer = False  # 是否使用DistributedOptimizer
overlap_param_gather = False  # 和DistibutedOptimizer一起使用，在forward时提前发起后面bucket的参数gather，实现计算和通信重叠
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # 根据命令行或者配置文件来覆盖参数
# 最终的配置文件
exp_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 判断是否使用了DDP
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run? 使用torchrun才会自动设置环境变量，即正常运行python文件不会开启ddp

# -----------------------------------------------------------------------------
# 不能一起使用的参数配置
assert (loss_reduction == "mean" and grad_scaling_before_comm) or (loss_reduction == "none" and not grad_scaling_before_comm),\
    "损失函数的reduction方式设置为None，必须和grad_scaling_before_comm=False配合使用，减少精度损失"

if dtype == "float16" and not grad_scaling_before_comm:
    raise ValueError("float16下不能在最后才进行梯度的缩放(not grad_scaling_before_comm)，因为可能会上溢")

if grad_div_total_tokens and use_powerSGD_hook:
    raise ValueError("PowerSGD和grad_div_total_tokens=True不能一起使用，")

if grad_buffer_is_powerSGD_error:
    assert use_powerSGD_hook and overlap_optim_step and overlap_zero_grad_buffer, "grad_buffer_is_powerSGD_error=True时，use_powerSGD_hook, overlap_optim_step和overlap_zero_grad_buffer都要为True"

assert use_reslog, "必须使用reslog，因为还会被用来判断文件是否保存成功"

if overlap_param_gather:
    assert use_distributed_optimizer, "需要和DistibutedOptimizer一起使用"

if not ddp:
    assert not use_bf16_compress_hook and not use_powerSGD_hook and not use_distributed_optimizer, "非DDP下不能使用bf16_compress_hook, powerSGD_hook, distributed_optimizer等"

if task_type == "dpo":
    assert grad_div_total_tokens is False, "dpo任务下grad_div_total_tokens必须为False"

# -----------------------------------------------------------------------------
# 设置ddp，判断是否是主进程
if ddp:
    process_group = init_process_group(backend=ddp_backend)  # 或者通过dist.group.WORLD来获得初始化后的process_group
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

if ddp:
    # 需要将out_dir广播到所有进程，因为时间可能会有点点差异
    object_list = [out_dir]
    torch.distributed.broadcast_object_list(object_list, src=0)  # 广播out_dir
    out_dir = object_list[0]

# -----------------------------------------------------------------------------
# 创建out_dir，并保存最终的配置文件信息
# 只有每个节点的local rank0需要创建out_dir（考虑powerSGD状态保存，每个rank有自己的powerSGD的error_dict状态）
if ddp_local_rank == 0:
    os.makedirs(out_dir, exist_ok=True)
if master_process:
    # 保存最终的配置文件信息
    save_run_exp_config(os.path.join(out_dir, "exp_config.py"), exp_config)

if ddp:
    torch.distributed.barrier()  # 等待所有进程都创建好out_dir

# -----------------------------------------------------------------------------
# 运行日志
# 创建logger，__name__表示运行文件名
# 如果存在log文件就删除
logger = logging.getLogger(__name__)
# 设置一些handle，如过滤、输出到文件等
logging.basicConfig(handlers=my_logging.get_all_handlers(out_dir), level=logging.INFO)
# 实验结果日志
if ddp_local_rank == 0 and use_reslog:
    # import wandb
    # wandb.init(project=wandb_project, name=wandb_run_name, config=exp_config)
    # wandb用起来有问题，改为自己的日志和画图工具
    reslog = ResLog(reslog_run_name, reslog_dir, reslog_save_interval)

# -----------------------------------------------------------------------------
# 每次迭代所训练的token数，2M = 16 * 8 * 8 * 2048
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * train_batch_size * max_seq_len
print_rank0(logger.info, f"tokens per iteration will be: {tokens_per_iter:,}")
print_rank0(logger.info, f"breaks down as: {gradient_accumulation_steps} grad_accum_steps * {ddp_world_size} processes * {train_batch_size} train_batch_size * {max_seq_len} max_seq_len")

# -----------------------------------------------------------------------------
# 设置随机种子
# torch.manual_seed(1337 + seed_offset)
torch.manual_seed(1337)  # 对于DDP来说，对每个进程应该设置相同的随机种子

# 允许tf32计算，比float16精度高，比float32速度快
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# -----------------------------------------------------------------------------
# 类型和上下文设置
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# float16下会使用GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# 设置自动混合精度的上下文
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# 初始化设置
print_rank0(logger.info, "Initializing model and optimizer")  # 通过这种方式可以避免在非master进程中打印
init_mode_optim_time = time.time()
# -----------------------------------------------------------------------------
# 实验过程中的信息
# iter_num = 0  # 放到上面作为参数了
best_val_loss = 1e9
# -----------------------------------------------------------------------------
# 模型初始化
model_config = Z2allConfig(**exp_config)  # resume也是用这个配置，不然还要用broadcast同步，有点麻烦
model = Z2allForCausalLM(model_config)
model.to(device)  # 将模型放到device上
# 保存模型配置和当前python文件
if master_process:
    # 保存模型配置，和相应配置类的代码文件
    model_config.save_pretrained(out_dir)  # 保存模型配置
    # 获取Z2allConfig所在的文件路径
    file_path = inspect.getfile(model_config.__class__)
    # 复制Z2allConfig所在的文件到out_dir
    shutil.copy(file_path, out_dir)
    # 将当前脚本文件复制到out_dir
    shutil.copy(os.path.abspath(__file__), out_dir)
# -----------------------------------------------------------------------------
# 梯度缩放器，优化器
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
init_mode_optim_time = time.time() - init_mode_optim_time
print_rank0(logger.info, f"Initialized  model and optimizer, {init_mode_optim_time:.4f}s")
# -----------------------------------------------------------------------------
# 使用自己的DDP和DistributedOptimizer包裹模型和优化器
# 只有在从头开始训练时才会进行包裹
if ddp:
    print_rank0(logger.info, f"Wrapping model into DDP container")
    
    warp_model_optim_time = time.time()
    
    model.bfloat16()  # 自己管理精度，这里可以设置为bfloat16
    
    # DDP
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        overlap_grad_reduce=overlap_grad_reduce,
        use_distributed_optimizer=use_distributed_optimizer,
        check_for_nan_in_grad=check_for_nan_in_grad,
        bucket_size=bucket_size)  # 默认bucket_size
    model = MyDDP(
        model,
        ddp_config,
        data_parallel_group=process_group,
        disable_bucketing=disable_bucketing)

    # Optimizer
    grad_scaling_factor = None
    if not grad_scaling_before_comm:
        # 是否在最后才进行grad_scaling，以减少精度损失（此时loss的损失函数的参数是None，而不是mean）
        grad_scaling_factor = 1.0 / tokens_per_iter
    optim_config = OptimizerConfig(
        precision_dtype=precision_dtype,
        grad_scaling_before_comm=grad_scaling_before_comm,
        grad_scaling_factor=grad_scaling_factor,
        overlap_optim_step=overlap_optim_step,
        overlap_zero_grad_buffer=overlap_zero_grad_buffer,
        grad_buffer_is_powerSGD_error=grad_buffer_is_powerSGD_error,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_param_gather=overlap_param_gather)
    optimizer = Float16OptimizerWithFloat16Params(optimizer, optim_config, model, scaler=scaler, grad_clip=grad_clip)
    
    warp_model_optim_time = time.time() - warp_model_optim_time
    
    print_rank0(logger.info, f"Wrapped  model into DDP container, {warp_model_optim_time:.4f}s")

# -----------------------------------------------------------------------------
# resume，加载模型参数和优化器状态等，先加载到rank0上，然后进行广播
powerSGD_state = None  # 看是否使用了PowerSGD
if resume:
    # 最好结果的前缀，或者最新, ["best", "last"]
    best1_prefix = f"{resume_from}1_"
    best2_prefix = f"{resume_from}2_"
    
    ckpt_out_dir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckpt_out_dir, exist_ok=True)
    
    print_rank0(logger.info, f"Resuming training from {out_dir} ({best1_prefix})")
    
    resume_time = time.time()
    
    # -----------------------------------------------------------------------------
    # 考虑保存时中断的特殊情况，假设：如果存在实验日志，则说明模型状态等文件保存成功了
    if ddp_local_rank == 0:
        # 没有中止的实验日志，说明还没有保存过模型状态等信息
        if not os.path.exists(os.path.join(ckpt_out_dir, best1_prefix + "reslog.pkl")) and not os.path.exists(os.path.join(ckpt_out_dir, best2_prefix + "reslog.pkl")):
            raise ValueError("没有中止的实验日志，说明还没有保存过模型状态等信息，建议从头开始训练")
        
        # best1文件不存在，说明之前的best1保存失败，若存在，说明文件损坏，删除
        if not os.path.exists(os.path.join(ckpt_out_dir, best1_prefix + "reslog.pkl")):
            # 删除ckpt_out_dir下所有best1前缀的文件
            for file_basename in os.listdir(ckpt_out_dir):
                if file_basename.startswith(best1_prefix):
                    os.remove(os.path.join(ckpt_out_dir, file_basename))
            # 然后将次优(best2)改名为最优(best1)
            for file_basename in os.listdir(ckpt_out_dir):
                if file_basename.startswith(best2_prefix):
                    new_file_basename = best1_prefix + file_basename[len(best2_prefix):]
                    os.rename(os.path.join(ckpt_out_dir, file_basename), os.path.join(ckpt_out_dir, new_file_basename))
    
    # -----------------------------------------------------------------------------
    # 加载模型状态
    # 待会广播给其他rank
    object_list = [None]
    if master_process:
        # 加载模型状态，先放在cpu上
        model_path = os.path.join(ckpt_out_dir, best1_prefix + "model.pt")
        model_state_dict = torch.load(model_path, map_location="cpu")
        object_list = [model_state_dict]
    # DDP下需要将状态广播到其他rank
    if ddp:
        torch.distributed.broadcast_object_list(object_list, src=0)  # 广播后放在cpu上
    model_state_dict = object_list[0]
    # 将对象中的所有张量复制到指定的设备上
    model_state_dict = copy_tensor_to_device_in_object(model_state_dict, device)
    # 加载状态到模型中
    model.load_state_dict(model_state_dict)
    # 即时释放内存
    object_list = None
    model_state_dict = None
    
    # -----------------------------------------------------------------------------
    # 加载其他状态，包括optimizer, iter_num, best_val_loss, powerSGD_state
    # 待会广播给其他rank
    object_list = [None]
    if master_process:
        ckpt_path = os.path.join(ckpt_out_dir, best1_prefix + "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        object_list = [checkpoint]
    # DDP下需要将状态广播到其他rank
    if ddp:
        torch.distributed.broadcast_object_list(object_list, src=0)  # 广播后放在cpu上
    checkpoint = object_list[0]
    
    # 获取RNG状态，在后面第一次测试后进行设置
    resume_cpu_rng_state = checkpoint.get("cpu_rng_state", None)
    resume_cuda_rng_state = checkpoint.get("cuda_rng_state", None)
    # 注意rng_state不能放到cuda上
    checkpoint.pop("cpu_rng_state", None)
    checkpoint.pop("cuda_rng_state", None)
    
    # 将对象中的所有张量复制到指定的设备上
    checkpoint = copy_tensor_to_device_in_object(checkpoint, device)
    
    # 加载状态到optimizer中
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # 训练时候的信息
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
    
    # 即时释放内存
    object_list = None
    checkpoint = None
    
    # 对于所有rank，加载其各自的powerSGD_state
    if ddp and use_powerSGD_hook:
        # 注意每个rank都有自己的powerSGD_state，文件名不同
        rank_prefix = best1_prefix + f"rank{ddp_rank}_"
        powerSGD_state_dict = torch.load(os.path.join(ckpt_out_dir, rank_prefix + "powerSGD_state.pt"), map_location=device)
        powerSGD_state = powerSGD_state_dict["powerSGD_state"]  # 后面作为powerSGD_hook的参数
        # 如果powerSGD_grad_buffer_is_error=True，则error_dict状态在grad_buffers中
        # 将其复制给对应的model的buffers的grad_data即可
        if grad_buffer_is_powerSGD_error:
            for buffer_idx, grad_buffer in enumerate(powerSGD_state_dict["grad_buffers"]):
                model.buffers[buffer_idx].grad_data.copy_(grad_buffer)
    
        # 注意，PowerSGD的状态，需要重新设置process_group，因为process_group不能被序列化
        # pytorch2.0后已经实现了__getstate__和__setstate__方法，可以直接序列化，里面包含了去除process_group的操作
        # parallel/distributed_data_parallel/ddp_comm_hooks/powerSGD_hook.py中添加了__getstate__和__setstate__方法
        if powerSGD_state is not None:
            powerSGD_state.process_group = process_group
            
        powerSGD_state_dict = None
    
    # 只有主进程需要加载实验过程日志
    if master_process:
        reslog.load(os.path.join(ckpt_out_dir, best1_prefix + "reslog.pkl"))  # 读取中止的实验日志
        
    resume_time = time.time() - resume_time
    print_rank0(logger.info, f"Resumed  training from {out_dir} ({best1_prefix}), {resume_time:.4f}s")

# 同步一下
if ddp:
    torch.distributed.barrier()

# -----------------------------------------------------------------------------
# 设置ddp com hook，看grad_scaling_before_comm设置来看是否要取world_size的平均值
# MyDDP中也实现了register_comm_hook，参数的位置和PyTorch的不太一样
# 基本的comm hook
if ddp:
    base_comm_hook = None
    if use_distributed_optimizer:
        base_comm_hook = reduce_scatter_hook
    else:
        base_comm_hook = all_reduce_hook
    cur_comm_hook = base_comm_hook
    comm_args = []  # 传入hook中的参数
    
    # 是否使用特其他的hook
    if use_powerSGD_hook:
        # 若没有resume，则初始化PowerSGDState
        if powerSGD_state is None:
            powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=matrix_approximation_rank,
                                           warm_start=warm_start, use_error_feedback=use_error_feedback, start_powerSGD_iter=start_powerSGD_iter,
                                           min_compression_rate=min_compression_rate, orthogonalization_epsilon=orthogonalization_epsilon,
                                           grad_buffer_is_powerSGD_error=grad_buffer_is_powerSGD_error,
                                           orthogonalize_in_float32=orthogonalize_in_float32,
                                           use_fixed_Q=use_fixed_Q)
        if use_bf16_compress_hook:
            cur_comm_hook = bf16_compress_wrapper(powerSGD_hook)
        else:
            cur_comm_hook = powerSGD_hook
        comm_args.append(powerSGD_state)
    elif use_bf16_compress_hook:
        comm_args.append(process_group)
        cur_comm_hook = bf16_compress_wrapper(base_comm_hook)
        
    # 是否overlap_optim_step
    if overlap_optim_step:
        cur_comm_hook = overlap_optim_step_wrapper(cur_comm_hook, optimizer)
        
    # 添加stream，用于异步通信
    cur_comm_hook = stream_wrapper(cur_comm_hook)
    
    # 最后对hook进行注册
    model.register_comm_hook(cur_comm_hook, *comm_args,
                             grad_scaling_factor=grad_scaling_factor, grad_scaling_before_comm=grad_scaling_before_comm)

# -----------------------------------------------------------------------------
# 任务构造器，用于生成训练和验证数据，不同进程会有不同的rng种子
# pretrian下已经预先进行了toeknize，所以不需要tokenizer
if task_type == "pretrain":
    tokenizer = None
elif task_type == "sft" or task_type == "dpo":
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
else:
    raise ValueError(f"Unknown task_type: {task_type}")
task_kwargs = dict(
    task_type=task_type,
    device=device,
    num_workers=num_workers,
    # same
    max_seq_len=max_seq_len,
    use_dataset_with_index=use_dataset_with_index,
    # sft, dpo
    tokenizer=tokenizer,
    user_name=user_name,
    assistant_name=assistant_name,
    # sft
    sft_type=sft_type,
    # dpo
    max_prompt_len=max_prompt_len,
    prompt_truncation_mode=prompt_truncation_mode,
    num_cpus=num_cpus,
)
if train_data_dir is not None and valid_data_dir is not None:
    # 按照文件夹划分好了数据集，每次取所有数据，valid_ratio设置为0
    task = {
        "train": Task(data_dir=train_data_dir, split="train", valid_ratio=0, batch_size=train_batch_size, **task_kwargs),
        "valid": Task(data_dir=valid_data_dir, split="valid", valid_ratio=0, batch_size=valid_batch_size, **task_kwargs),
    }
else:
    # 只有一个文件夹，dataset类中自动进行划分
    task = {
        "train": Task(data_dir=data_dir, split="train", valid_ratio=valid_ratio, batch_size=train_batch_size, **task_kwargs),
        "valid": Task(data_dir=data_dir, split="valid", valid_ratio=valid_ratio, batch_size=valid_batch_size, **task_kwargs),
    }

# -----------------------------------------------------------------------------
# 根据创建的数据集，计算一个epoch的iter数
train_num_samples_per_epoch = task["train"].num_samples  # 一个epoch总共的样本数
global_train_batch_size = train_batch_size * gradient_accumulation_steps * ddp_world_size  # 全局的train_batch_size
train_num_iters_per_epoch = train_num_samples_per_epoch // global_train_batch_size  # 一个epoch总共的iter数
# 计算本次训练需要训练的iter数
max_iters = epoch * train_num_iters_per_epoch
# 设置学习率衰减策略参数
lr_decay_iters = max_iters * 0.9  # should be ~= max_iters per Chinchilla 开始停止学习率衰减的step
min_lr = learning_rate / 10  # 衰减到的最小学习率，~= learning_rate/10 per Chinchilla
warmup_iters = int(warmup_ratio * max_iters)  # warmup的step数

# -----------------------------------------------------------------------------
# 学习率衰减策略 (cosine with warmup)
def get_lr(it):
    global learning_rate, warmup_iters, lr_decay_iters, min_lr
    
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)  # 从0到1
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1 从1到0
    return min_lr + coeff * (learning_rate - min_lr)

# -----------------------------------------------------------------------------
# 测试函数，只在rank0上进行验证
@torch.no_grad()
def estimate_loss():
    global dpo_config
    
    # 计算valid数据集需要验证的样本数
    valid_num_samples_per_epoch = task["valid"].num_samples  # 一个epoch总共的样本数
    valid_num_samples = valid_iters * valid_batch_size if valid_iters is not None else valid_num_samples_per_epoch  # 设置下的要验证的样本数
    
    assert valid_num_samples_per_epoch is not None or valid_num_samples is not None, "PretokDataset下，valid_iters必须设置，因为构建方法和index不太一样，不知道总共有多少个sample"
    
    # 应该验证的样本数
    cur_valid_num_samples = None
    if valid_num_samples_per_epoch is not None and valid_num_samples is not None:
        cur_valid_num_samples = min(valid_num_samples_per_epoch, valid_num_samples)
    elif valid_num_samples_per_epoch is not None:
        cur_valid_num_samples = valid_num_samples_per_epoch
    elif valid_num_samples is not None:
        cur_valid_num_samples = valid_num_samples

    out = {}
    model.eval()
    for split in ["train", "valid"]:
        # 根据要验证的样本数，确定需要验证的iter数
        # 即train和valid验证的样本数一样，向下取整，至少验证一个iter
        cur_valid_iters = cur_valid_num_samples // task[split].batch_size

        assert cur_valid_iters > 0, f"valid样本太少，一个valid_batch_size都不能构建"
        
        print_rank0(logger.info, f"Estimating loss, split={split}, {split}_batch_size={task[split].batch_size}, num_iters={cur_valid_iters}")
        
        # task任务也需要转换到eval状态
        # 因为在DDP下，train会受到影响，导致取得batch不是按顺序取的，而是按ddp_world_size跳着取
        # 设置的结果：  step 0: train loss 11.4743, valid loss 11.4742
        # 不设置的结果：step 0: train loss 11.4735, valid loss 11.4742
        # 所以证明了train在ddp下会受到影响，严谨点，应该设置，记得恢复
        task[split].eval()
        
        batch_iter = task[split].iter_batches()
        losses = torch.zeros(cur_valid_iters)  # keep on CPU
        for k in range(cur_valid_iters):
            valid_batch = next(batch_iter)
            with ctx:
                loss = forward_step(model, valid_batch, task_type=task_type, dpo_config=dpo_config)
            if loss_reduction == "none":
                loss = torch.mean(loss.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
        
        # task任务恢复为原来的模式，即设置回split
        task[split].restore_mode()
        
    model.train()
    return out

# -----------------------------------------------------------------------------
# 准备训练集，跳过训练过的样本
skip_batches_per_device = int(iter_num * gradient_accumulation_steps * skip_scaling_factor)  # 每个设备跳过的batch数
skip_data_time = time.time()
print_rank0(logger.info, f"Skipping {iter_num} iters ({skip_batches_per_device} skip_batches_per_device)")

train_batch_iter = task["train"].iter_batches(skip_batches=skip_batches_per_device)
train_batch = next(train_batch_iter)  # 在里面跳过skip_batches

print_rank0(logger.info, f"Skipped  {iter_num} iters ({skip_batches_per_device} skip_batches_per_device), {time.time() - skip_data_time:.4f}s")

# 同步一下
if ddp:
    torch.distributed.barrier()

# -----------------------------------------------------------------------------
# 开始训练
train_time0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
# dpo_config
dpo_config = None
if task_type == "dpo":
    if use_peft:
        ref_model = None
    else:
        ref_model = create_reference_model(raw_model).to(device)
    dpo_config = DPOConfig(
        ref_model=ref_model,
        use_peft=use_peft,
        pad_token_id=tokenizer.pad_token_id,
        ignore_index=ignore_index,
        loss_type=dpo_loss_type,
        beta=dpo_beta,
        label_smoothing=dpo_label_smoothing,
        reference_free=reference_free,
    )
# mfu
running_mfu = -1.0
print_rank0(logger.info, f"Start training loop")
while True:
    # -----------------------------------------------------------------------------
    # 根据iter，调整学习率
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num + 1) if decay_lr else learning_rate  # 从1开始，要不然第一个step的lr是0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        
    if resume:
        # 只会设置一次
        # resume后的第一次验证后，需要设置回之前训练的RNG状态
        torch.set_rng_state(resume_cpu_rng_state)
        torch.cuda.set_rng_state(resume_cuda_rng_state)
        resume_cpu_rng_state = None
        resume_cuda_rng_state = None
    
    # -----------------------------------------------------------------------------
    # 验证，只在rank0上验证和保存
    # 但是其他rank也需要知道该信息，用于保存powerSGD的状态（每个rank都需要保存自己的error_dict）
    valid_loss = torch.tensor([-1.0], dtype=torch.float32, device=device)  # 初始为-1，若验证过了，则必为正数，这样可以判断是否验证过了
    if iter_num % valid_interval == 0 and master_process:
        valid_t0 = time.time()
        losses = estimate_loss()
        valid_dt = time.time() - valid_t0
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}, {valid_dt:.4f}s")
        if use_reslog and master_process:
            reslog.log({
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,
                "loss/train": losses["train"],
                "loss/valid": losses["valid"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }, name="valid", step = iter_num)
            
        valid_loss = torch.tensor(losses["valid"], dtype=torch.float32, device=device)
        
    # 将valid_loss广播给其他rank
    if iter_num % valid_interval == 0 and ddp:
        torch.distributed.broadcast(valid_loss, src=0, async_op=False)
        
    valid_loss = valid_loss.item()  # 没有验证的话，则值为-1.0
    
    # 保存最新状态，第一次iter不保存，resume后的第一次验证也不保存
    if iter_num % valid_interval == 0 and iter_num > 0 and not resume:
        save_checkpoint_time = time.time()
        
        # 保存状态
        save_checkpoint(globals(), prefix="last")
            
        save_checkpoint_time = time.time() - save_checkpoint_time
        print_rank0(logger.info, f"save last checkpoint to {out_dir}, {save_checkpoint_time:.4f}s")
        
    # -----------------------------------------------------------------------------
    # 看看是否需要保存最优checkpoint，resume后的第一个不需要保存，因为还是原来的
    if iter_num % valid_interval == 0 and valid_loss < best_val_loss and not resume:
        best_val_loss = valid_loss
        if iter_num > 0:
            save_checkpoint_time = time.time()
            
            # 保存状态
            save_checkpoint(globals(), prefix="best")
                
            save_checkpoint_time = time.time() - save_checkpoint_time
            print_rank0(logger.info, f"save best checkpoint to {out_dir}, {save_checkpoint_time:.4f}s")
    if iter_num % valid_interval == 0:
        # 验证过了，重置下训练的开始时间
        train_time0 = time.time()
    resume = False  # resume后的第一个不需要保存，因为还是原来的
    
    if iter_num == 0 and valid_only:
        break
    
    # 同步一下，防止验证太久，其他进行完成forward后，会在backward时等待通信，等待时间过长可能会报错
    if ddp:
        torch.distributed.barrier()
    
    # 记录每个micro所耗费的时间
    micro_times = []
    
    # 保存loss，用于log
    train_loss = torch.tensor([0.0], device=device)
    
    # -----------------------------------------------------------------------------
    # 前向传播和反向传播，梯度更新
    # 使用model.no_sync()来设置是否同步
    no_sync = nullcontext
    if ddp:
        no_sync = model.no_sync
    with no_sync():
        for micro_step in range(gradient_accumulation_steps - 1):
            micro_time = time.time()  #! 1
            with ctx:
                loss = forward_step(model, train_batch, task_type=task_type, dpo_config=dpo_config)
                if loss_reduction == "mean":
                    loss = loss / gradient_accumulation_steps
                else:
                    # 否则为"none"，则grad在optim.step中进行scale，减少精度损失
                    loss = torch.sum(loss.view(-1))
            train_loss += loss.clone().detach()
            # 立刻异步预取下一个batch的数据，与forward并行
            train_batch = next(train_batch_iter)
            # scaler和反向传播
            # overlap_grad_reduce时会自动进行梯度的all-reduce，并且取所有的word_size的平均值
            scaler.scale(loss).backward()
    
            # 同步以获取真实的耗时
            if ddp and sync_for_true_time:
                torch.distributed.barrier()
                
            micro_time = time.time() - micro_time  #! 1
            micro_times.append(micro_time)
    
    last_micro_time = time.time()  #! 2
    
    # last_microbatch，需要同步了，backward中会进行梯度的通信（overlap_optim_step下还会进行optim.step()）
    with ctx:
        loss = forward_step(model, train_batch, task_type=task_type, dpo_config=dpo_config)
        if loss_reduction == "mean":
            loss = loss / gradient_accumulation_steps
        else:
            # 否则为"none"，则grad在optim.step中进行scale，减少精度损失
            loss = torch.sum(loss.view(-1))
    train_loss += loss.clone().detach()
    # 立刻异步预取下一个batch的数据，与forward并行
    train_batch = next(train_batch_iter)
    # scaler和反向传播
    # overlap_grad_reduce时会自动进行梯度的all-reduce，并且取所有的word_size的平均值
    scaler.scale(loss).backward()
    
    optim_step_time = time.time()  #! 3
    
    optimizer.step()  # scaler和grad_clip放在了这里面，里面会进行参数更新的同步
    optimizer.zero_grad()  # overlap_param_gather时会在这里发起all-gather同步
    
    # 同步一下
    if ddp:
        torch.distributed.barrier()

    last_micro_time = time.time() - last_micro_time  #! 2
    micro_times.append(last_micro_time)
    optim_step_time = time.time() - optim_step_time  #! 3
    
    # 获取所有rank下的loss
    if ddp:
        torch.distributed.all_reduce(train_loss, group=process_group, async_op=False)
    # 需要取平均值
    if loss_reduction == "mean":
        train_loss = train_loss / ddp_world_size
    else:
        train_loss = train_loss / tokens_per_iter
    
    # -----------------------------------------------------------------------------
    # 输出结果
    # timing and logging
    train_time1 = time.time()
    dt = train_time1 - train_time0
    train_time0 = train_time1
    if iter_num % log_interval == 0 and master_process:
        # 调用.item()方法会导致CPU等待GPU计算完成，因为需要将数据从GPU内存复制到CPU内存。
        lossf = train_loss.item()
        if local_iter_num >= 5:  # 过几个iter再计算mfu（Model FLOPs Utilization）
            num_samples = train_batch_size * gradient_accumulation_steps
            
            # 注意sft和dpo下，每个样本的最大长度不一定为max_seq_len，所以mfu计算出来的不准
            if task_type == "dpo":
                num_samples *= 2  # dpo下每个样本都会进行两次forward，因为有chosen和rejected两个
                
            mfu = estimate_mfu(raw_model, num_samples, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            # 前几个step不准，因为模型还没有稳定下来
            # 防止不准的数值对坐标轴的影响
            # 同时不保存测试时训练的实验结果，因为时间计算的是测试+训练的时间
            if use_reslog and master_process and iter_num % valid_interval != 0:
                reslog.log({
                    "iter": iter_num,
                    "tokens": iter_num * tokens_per_iter,
                    "loss": lossf,
                    "dt": dt,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }, name="train", step=iter_num)
        print_rank0(logger.info, 
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt:.4f}s | mfu {running_mfu*100:.2f}% | micro_time0: {micro_times[0]:.4f}s | micro_time1: {micro_times[1]:.4f}s | last_micro_time: {micro_times[-1]:.4f}s | optim_step_time: {optim_step_time:.4f}s"
        )
    iter_num += 1
    local_iter_num += 1

    # 中止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # gloo退出有问题，这行代码不会退出
