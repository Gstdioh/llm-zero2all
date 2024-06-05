"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m pretrain.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 pretrain.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 pretrain.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 pretrain.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
$ export NCCL_IB_DISABLE=1
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 pretrain.py

my_run
# single
$ python pretrain.py --batch_size=2 --gradient_accumulation_steps=2
# 看速度，不用flash可能会超显存，所以使用小的batch_size
$ python pretrain.py --batch_size=2 --gradient_accumulation_steps=16
# 看显存占用
$ python pretrain.py --batch_size=16 --gradient_accumulation_steps=2

OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain_my_ddp.py --gradient_accumulation_steps=12

# gpu4
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py
# test
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=gloo  # gloo
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=nccl  # nccl
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 NCCL_BUFFLE_SIZE=16777216 torchrun --standalone --nproc_per_node=4 pretrain.py
$ OMP_NUM_THREADS=8 NCCL_BUFFLE_SIZE=16777216 NCCL_P2P_LEVEL=5 torchrun --standalone --nproc_per_node=4 pretrain.py # error

# gpu4, gpu4_2
- gpu4
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain_my_ddp.py

- gpu4_2
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain_my_ddp.py
"""

import math
import os
import inspect
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
import torch.distributed
from transformers import AutoConfig, AutoTokenizer

from my_dataset import Task
from model import Z2allConfig, Z2allForCausalLM
import utils
from utils import get_logger, estimate_mfu, configure_optimizers, ResLog, save_run_exp_config

from parallel.distributed_data_parallel import DistributedDataParallelConfig
from parallel.distributed_data_parallel import DistributedDataParallel as MyDDP
from optimizer import OptimizerConfig, FP32Optimizer, Float16OptimizerWithFloat16Params
from parallel.distributed_optimizer import DistributedOptimizer
from parallel.distributed_data_parallel.ddp_comm_hooks.default_hooks import all_reduce_hook, reduce_scatter_hook, bf16_compress_wrapper, stream_wrapper
from parallel.distributed_data_parallel.ddp_comm_hooks.overlap_optim_step_hooks import overlap_optim_step_wrapper
from parallel.distributed_data_parallel.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook


os.environ["NCCL_IB_DISABLE"] = "1"  # disable infiniband
os.environ["NCCL_IBEXT_DISABLE"] = "1"  # 有用啦!
os.environ["NCCL_P2P_DISABLE"] = "1"  # disable p2p
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of threads for OpenMP
# os.environ["NCCL_DEBUG"] = "WARN"  # set NCCL debug level, ["WARN", "INFO"]，用于测试

# os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # 设置每个 CUDA 设备的最大并行内核执行数，速度还快了？


# 用于torch.compile，需要PyTorch>=2.0
if torch.__version__ >= "2.0.0":
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128  # 原来是64，有警告，设大点加快编译

# 用于找到pad的id
# tokenizer = AutoTokenizer.from_pretrained("tokenizer/hf_bbpe_tokenizer", trust_remote_code=True)

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
eval_interval = 200  # 每eval_interval个step验证一次，这里设置大点（先不测试，因为我还没测试好MyDDP的保存）
log_interval = 1
eval_iters = 100  # 每次验证的step数
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
resume = False  # if True, resume training from the last checkpoint
sync_for_true_micro_time = False  # 是否同步以获取micro真实耗费的时间，测试下可以用
# my logging
use_reslog = True  # wandb用起来有问题，改为自己的日志和画图工具
reslog_dir = "reslog"
reslog_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
reslog_save_interval = 10  # 想快速看结果，可以用小点的数
# data
train_bin_dir = "data/02_train_data_more/01_bin_for_train_hf"
valid_bin_dir = "data/02_train_data_more/02_bin_for_valid_hf"
num_workers = 0  # 数据加载器的工作进程数
## global_batch_size=batch_size*gradient_accumulation_steps*ddp_world_size
batch_size = 8  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
grad_div_total_tokens = False  # 是否在计算梯度时除以总的token数，设置reduction="none" and grad_scaling_before_comm=False，使用PowerSGD时不能使用（loss不好，可能因为PowerSGD对数大的压缩不好，有正交化操作）
# model
vocab_size = 64320  # 实际是64012个，写大点方便扩展，注意最好是8的倍数，见指导：https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#tc-guidelines-padding
hidden_dim = 2048
intermediate_size = 5632
n_layers = 28
n_heads = 16
n_kv_heads = 8  # 用于GQA
max_seq_len = max_seq_len
initializer_range = 0.02  # 参数初始化时的标准差
rms_norm_eps = 1e-5  # 防止除0的小数
pad_token_id = 64006  # tokenizer.special_tokens["<|PAD|>"]  # pad token 64006
tie_word_embeddings = False  # 是否共享word embedding和word prediction的参数
rope_theta = 10000.0
rope_scaling = None  # 缩放方法，用于长度外推
attention_bias = True  # attention中的project是否加bias，Qwen中加了
attention_dropout = 0.1  # TODO: 或许不用设置dropout
dropout1 = 0.1
dropout2 = 0.1
residual_in_fp32 = True  # 残差连接是否使用fp32
loss_reduction = "none" if grad_div_total_tokens else "mean"  # 损失函数的reduction方式，"mean" or "none"，使用"none"可以和grad_scaling_before_comm=False配合使用，减少精度损失
# adamw optimizer
## gradient_accumulation_steps=gradient_accumulation_steps*ddp_world_size
gradient_accumulation_steps = 128  # used to simulate larger batch sizes
learning_rate = 3e-4  # max learning rate，参考Qwen
max_iters = 100000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False  # use PyTorch 2.0 to compile the model to be faster
# 分布式配置
# 通信后端
# 见：https://huggingface.co/docs/transformers/perf_train_gpu_many
# 使用nccl时，如果没有nvlink，则需要设置NCCL_P2P_DISABLE=1
# 没有nvlink时，在单节点下DP比DDP更快，但是DP不支持多节点训练
# 因为我的环境没有nvlink，所以我使用的是gloo后端
# 但是gloo又与hook有问题，还是用nccl吧
# 1. gloo，不支持bfloat16，使用PowerSGD时会卡住，强行退出时GPU不会立即释放
# 2. nccl，需要设置NCCL_IB_DISABLE=1，NCCL_IBEXT_DISABLE=1，NCCL_P2P_DISABLE=1
ddp_backend = "nccl"  # ddp backend, can be 'nccl', 'gloo'
# 梯度通信优化
use_bf16_compress_hook = False
use_powerSGD_hook = True
# DistributedDataParallelConfig
grad_reduce_in_fp32 = False
overlap_grad_reduce = True
use_distributed_optimizer = False
check_for_nan_in_grad = False
bucket_size = 10_000_000
disable_bucketing = False
# OptimizerConfig
precision_dtype = dtype
grad_scaling_before_comm = False if grad_div_total_tokens else True  # 是否在通信前进行梯度缩放，建议bfloat16下设为False，在最后除以值，减少精度损失
overlap_optim_step = True
overlap_zero_grad_buffer = True
use_distributed_optimizer = False
overlap_param_gather = False
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # 根据命令行或者配置文件来覆盖参数
# 最终的配置文件
exp_config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# 删除tokenizer，后面不会用到了
tokenizer = None

# -----------------------------------------------------------------------------
# 不能一起使用的参数配置
assert (loss_reduction == "mean" and grad_scaling_before_comm) or (loss_reduction == "none" and not grad_scaling_before_comm),\
    "损失函数的reduction方式设置为None，必须和grad_scaling_before_comm=False配合使用，减少精度损失"

if dtype == "float16" and not grad_scaling_before_comm:
    raise ValueError("float16下不能在最后才进行梯度的缩放(not grad_scaling_before_comm)，因为可能会上溢")

if grad_div_total_tokens and use_powerSGD_hook:
    raise ValueError("PowerSGD和grad_div_total_tokens=True不能一起使用，")

# -----------------------------------------------------------------------------
# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla 开始停止学习率衰减的step
# min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
min_lr = learning_rate / 10  # 衰减到的最小学习率

# -----------------------------------------------------------------------------
# 设置ddp，判断是否是主进程
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run? 使用torchrun才会自动设置环境变量，即正常运行python文件不会开启ddp
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
    ddp_world_size = 1
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    # 保存最终的配置文件信息
    save_run_exp_config(os.path.join(out_dir, "exp_config.py"), exp_config)

# -----------------------------------------------------------------------------
# 运行日志
# 创建logger，__name__表示运行文件名
# 如果存在log文件就删除
logger = None
if master_process:
    log_path = os.path.join(out_dir, 'info.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logger = get_logger(log_dir=out_dir, name=__name__, log_filename='info.log', level="INFO")
# 实验结果日志
if use_reslog and master_process:
    # import wandb
    # wandb.init(project=wandb_project, name=wandb_run_name, config=exp_config)
    # wandb用起来有问题，改为自己的日志和画图工具
    reslog = ResLog(reslog_run_name, reslog_dir, reslog_save_interval)

# 每次迭代所训练的token数，1M = 1 * 4 * 128 * 2048
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    logger.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    logger.info(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

# -----------------------------------------------------------------------------
# 设置随机种子
torch.manual_seed(1337 + seed_offset)

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# -----------------------------------------------------------------------------
# 类型和上下文设置
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# 设置自动混合精度的上下文
ctx = (
    nullcontext()
    if device_type == "cpu"
    # else torch.amp.autocast(device_type=device_type, dtype=ptdtype)  # 原来的代码
    else torch.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# 任务构造器，用于生成训练和验证数据
# task-specific setup
iter_batches = partial(
    Task.iter_batches,
    batch_size=batch_size,
    max_seq_len=max_seq_len,
    train_bin_dir=train_bin_dir,
    valid_bin_dir=valid_bin_dir,
    device=device,
    num_workers=0,
)

# -----------------------------------------------------------------------------
# 模型初始化，配置初始化，配置DDP
# init these up here, can override if resume = True (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# model init
powerSGD_state = None  # 看是否使用了PowerSGD
if not resume:
    # init a new model from scratch
    _ = logger.info("Initializing a new model from scratch") if master_process else None  # 通过这种方式可以避免在非master进程中打印
    model_config = Z2allConfig(**exp_config)
    model = Z2allForCausalLM(model_config)
    if master_process:
        # 保存模型配置，和相应配置类的代码文件
        model_config.save_pretrained(out_dir)  # 保存模型配置
        # 获取Z2allConfig所在的文件路径
        file_path = inspect.getfile(model_config.__class__)
        # 复制Z2allConfig所在的文件到out_dir
        shutil.copy(file_path, out_dir)
else:  # resume
    _ = logger.info(f"Resuming training from {out_dir}") if master_process else None
    
    best1_prefix = "best1_"
    
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, best1_prefix + "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 获取模型配置
    try:
        model_config = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    except:
        model_config = Z2allConfig(**exp_config)
    # 构建模型
    model = Z2allForCausalLM(model_config)
    
    # 加载模型状态，到特定的device
    state_dict = torch.load(os.path.join(out_dir, best1_prefix + "model.pt"), map_location=device)
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # 使用了torch.compile才会出现这个问题，即前缀会有"_orig_mod."
    wanted_prefix = ""
    unwanted_prefix = "_orig_mod."
    full_prefix = wanted_prefix + unwanted_prefix
    for k, v in list(state_dict.items()):
        if k.startswith(full_prefix):
            new_k = wanted_prefix + k[len(full_prefix) :]
            state_dict[new_k] = state_dict.pop(k)  # 修改键名
    # 加载模型状态
    model.load_state_dict(state_dict)
    state_dict = None  # free up memory
    
    # PowerSGD的状态，需要重新设置process_group，因为process_group不能被序列化
    # pytorch2.0后已经实现了__getstate__和__setstate__方法，可以直接序列化，里面包含了去除process_group的操作
    powerSGD_state = checkpoint.get("powerSGD_state", None)
    if powerSGD_state is not None:
        powerSGD_state.process_group = process_group
    
    # 只有主进程需要加载实验过程日志
    if master_process:
        reslog.load(os.path.join(out_dir, best1_prefix + "reslog.pkl"))  # 读取中止的实验日志
        
    # 训练时候的信息
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# -----------------------------------------------------------------------------
# 训练所需，梯度缩放器，优化器
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type, logger, master_process)
# 非ddp下，在这里载入优化器状态，否则在后面载入
if resume and "optimizer" in checkpoint and not ddp:
    optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

# -----------------------------------------------------------------------------
# 学习率衰减策略
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
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
# 测试函数
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                model_outputs = model(X, Y)
                loss = model_outputs["loss"]
            if loss_reduction == "none":
                loss = torch.mean(loss.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -----------------------------------------------------------------------------
# wrap模型
# 编译模型，需要PyTorch>=2.0
if compile and torch.__version__ >= "2.0":
    if master_process:
        logger.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
# -----------------------------------------------------------------------------
# 使用自己的DDP和DistributedOptimizer包裹模型和优化器
# wrap model into DDP container, 只有在从头开始训练时才会进行包裹
if ddp:
    _ = logger.info(f"wrapping model into DDP container") if master_process else None
    
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
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_param_gather=overlap_param_gather)
    optimizer = Float16OptimizerWithFloat16Params(optimizer, optim_config, model, scaler=scaler, grad_clip=grad_clip)

    # resume，加载优化器状态
    if resume and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

# -----------------------------------------------------------------------------
# 准备训练集
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
# 如果resume，需要跳过前面的iter
if resume:
    skip_data_time = time.time()
    for _ in range(iter_num):
        X, Y = next(train_batch_iter)
    _ = logger.info(f"skip {iter_num} iters time: {time.time() - skip_data_time:.4f}s") if master_process else None

# -----------------------------------------------------------------------------
# 预热
# 由于DDP下的bucket可能要重新构建，需要预热下，重建需要2个iter，情况:
# 注意考虑PowerSGDState的start_powerSGD_iter应该变化，可能需要临时禁用powerSGD_hook
# 预热时不使用powerSGD_hook，为了让模型的bucket构建好，因为第一个iter后bucket会被rebuilt
# 1. 在用torch的ZeroRedundancyOptimizer时，可能更需要预热，可能前几个iter不会更新参数
# 2. resume情况下，因为resume后，DDP又要重新开始构建bucket了
# 见：powerSGD_hook.py的注释
# 见：https://github.com/pytorch/pytorch/pull/73732
#* 自己的DDP不需要预热，构建DDP的时候已经确定好bucket了
#* overlap_optim_step时会在backward中进行更新，所以不能这样预热
# if ddp:
#     warm_for_bucket_rebuilt_time = time.time()
#     for _ in range(1):
#         with ctx:
#             model_outputs = model(X, Y)
#             loss = model_outputs["loss"]
#             loss = loss / gradient_accumulation_steps
#         scaler.scale(loss).backward()
#         optimizer.zero_grad(set_to_none=True)
#     torch.cuda.synchronize()
#     torch.distributed.barrier()
#     _ = logger.info(f"warm_for_bucket_rebuilt_time: {time.time() - warm_for_bucket_rebuilt_time:.4f}s") if master_process else None

# -----------------------------------------------------------------------------
# 预热后，设置ddp com hook，还是会取world_size的平均值
# MyDDP中也实现了register_comm_hook，不过参数的位置不太一样
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
            powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
                                           warm_start=True, use_error_feedback=True, start_powerSGD_iter=2,
                                           min_compression_rate=2, orthogonalization_epsilon=1e-6)
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

# 同步下
if ddp:
    torch.cuda.synchronize()
    torch.distributed.barrier()

# -----------------------------------------------------------------------------
# 开始训练
train_time0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
_ = logger.info(f"start training loop") if master_process else None
while True:
    # 根据iter，调整学习率
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num + 1) if decay_lr else learning_rate  # 从1开始，要不然第一个step的lr是0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # 验证
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        val_t0 = time.time()
        losses = estimate_loss()
        val_dt = time.time() - val_t0
        logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, {val_dt:.4f}s")
        if use_reslog and master_process:
            reslog.log({
                "iter": iter_num,
                "tokens": iter_num * tokens_per_iter,
                "loss/train": losses["train"],
                "loss/val": losses["val"],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            }, name="valid", step = iter_num)
            
        # 保存checkpoint，resume后的第一个不需要保存，因为还是原来的
        if (losses["val"] < best_val_loss or always_save_checkpoint) and not resume:
            best_val_loss = losses["val"]
            if iter_num > 0:
                # 保存PowerSGD的状态
                if use_powerSGD_hook:
                    # 保存process_group有问题：https://discuss.pytorch.org/t/how-to-resume-with-powersgd-enabled-training/148747/2
                    pg = powerSGD_state.process_group
                    powerSGD_state.process_group = None
                
                # 为了防止保存最优时程序中断，即最优保存失败，需要保存一个次优版本
                # 先将次优(best2)删除，然后将最优(best1)改名为次优
                best1_prefix = "best1_"
                best2_prefix = "best2_"
                # 先将out_dir下的次优前缀文件删除
                for file_basename in os.listdir(out_dir):
                    if file_basename.startswith(best2_prefix):
                        os.remove(os.path.join(out_dir, file_basename))
                # 然后将最优(best1)改名为次优
                for file_basename in os.listdir(out_dir):
                    if file_basename.startswith(best1_prefix):
                        new_file_basename = best2_prefix + file_basename[len(best1_prefix):]
                        os.rename(os.path.join(out_dir, file_basename), os.path.join(out_dir, new_file_basename))
                
                # 保存完次优后，可以放心进行保存最优了
                # 1. 单独保存模型权重文件
                torch.save(raw_model.state_dict(), os.path.join(out_dir, best1_prefix + "model.pt"))
                # 2. 保存训练状态
                checkpoint = {
                    # "model": model_save_method,
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "powerSGD_state": powerSGD_state,
                }
                torch.save(checkpoint, os.path.join(out_dir, best1_prefix + "ckpt.pt"))
                # 3. 保存实验过程日志，可用resplot来展示
                reslog.save(os.path.join(out_dir, best1_prefix + "reslog.pkl"))
                
                # model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)  # TODO
                
                logger.info(f"save checkpoint to {out_dir}")
                
                # 记得还原PowerSGD的状态
                if use_powerSGD_hook:
                    # 保存process_group有问题：https://discuss.pytorch.org/t/how-to-resume-with-powersgd-enabled-training/148747/2
                    powerSGD_state.process_group = pg
        train_time0 = time.time()  # 因为经过了测试，所以训练的起始时间需要重新设置
    resume = False  # resume后的第一个不需要保存，因为还是原来的
    if iter_num == 0 and eval_only:
        break
    
    # 同步以获取真实的耗时
    if ddp and sync_for_true_micro_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()
    
    # 记录每个micro所耗费的时间
    micro_times = []
    
    # 保存loss，用于log
    train_loss = torch.tensor([0.0], device=device)
    
    # 前向传播和反向传播，梯度更新
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    # 使用model.no_sync()来设置是否同步
    with model.no_sync():
        for micro_step in range(gradient_accumulation_steps - 1):
            micro_time = time.time()  #! 1
            with ctx:
                model_outputs = model(X, Y)
                loss = model_outputs["loss"]
                if loss_reduction == "mean":
                    loss = loss / gradient_accumulation_steps
                else:
                    # 否则为"none"，则grad在optim.step中进行scale，减少精度损失
                    loss = torch.sum(loss.view(-1))
            train_loss += loss.clone().detach()
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()  # 同步的时候会自动进行梯度的all-reduce，并且取所有的word_size的平均值
    
            # 同步以获取真实的耗时
            if ddp and sync_for_true_micro_time:
                torch.cuda.synchronize()
                torch.distributed.barrier()
                
            micro_time = time.time() - micro_time  #! 1
            micro_times.append(micro_time)
    
    last_micro_time = time.time()  #! 2
    
    # last_microbatch，需要同步了，backward中会进行梯度的通信（overlap_optim_step下还会进行optim.step()）
    with ctx:
        model_outputs = model(X, Y)
        loss = model_outputs["loss"]
        if loss_reduction == "mean":
            loss = loss / gradient_accumulation_steps
        else:
            # 否则为"none"，则grad在optim.step中进行scale，减少精度损失
            loss = torch.sum(loss.view(-1))
    train_loss += loss.clone().detach()
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = next(train_batch_iter)
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()  # 同步的时候会自动进行梯度的all-reduce，并且取所有的word_size的平均值
    
    optim_step_time = time.time()  #! 3
    
    optimizer.step()  # scaler和grad_clip放在了这里面
    optimizer.zero_grad(set_to_none=True)
    
    # 同步以获取真实的耗时
    if ddp and sync_for_true_micro_time:
        torch.cuda.synchronize()
        torch.distributed.barrier()

    last_micro_time = time.time() - last_micro_time  #! 2
    micro_times.append(last_micro_time)
    optim_step_time = time.time() - optim_step_time  #! 3
    
    torch.distributed.all_reduce(train_loss, group=process_group, async_op=False)
    # 需要取平均值
    if loss_reduction == "mean":
        train_loss = train_loss / ddp_world_size
    else:
        train_loss = train_loss / tokens_per_iter
    
    # 输出结果
    # timing and logging
    train_time1 = time.time()
    dt = train_time1 - train_time0
    train_time0 = train_time1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        # 调用.item()方法会导致CPU等待GPU计算完成，因为需要将数据从GPU内存复制到CPU内存。
        lossf = train_loss.item()
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = estimate_mfu(raw_model, batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            # 前几个step不准，因为模型还没有稳定下来
            # 防止不准的数值对坐标轴的影响
            # 同时不保存测试时训练的实验结果，因为时间计算的是测试+训练的时间
            if use_reslog and master_process and iter_num % eval_interval != 0:
                reslog.log({
                    "iter": iter_num,
                    "tokens": iter_num * tokens_per_iter,
                    "loss": lossf,
                    "dt": dt,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }, name="train", step=iter_num)
        logger.info(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt:.4f}s | mfu {running_mfu*100:.2f}% | micro_time0: {micro_times[0]:.4f}s | micro_time1: {micro_times[1]:.4f}s | last_micro_time: {micro_times[-1]:.4f}s | optim_step_time: {optim_step_time:.4f}s"
        )
    iter_num += 1
    local_iter_num += 1

    # 中止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # gloo退出有问题，这行代码不会退出
