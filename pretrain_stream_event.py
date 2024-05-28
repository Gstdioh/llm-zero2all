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

OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain.py

# gpu4
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py
# test
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=gloo  # gloo
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=nccl  # nccl
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 NCCL_BUFFLE_SIZE=16777216 torchrun --standalone --nproc_per_node=4 pretrain.py
$ OMP_NUM_THREADS=8 NCCL_BUFFLE_SIZE=16777216 NCCL_P2P_LEVEL=5 torchrun --standalone --nproc_per_node=4 pretrain.py # error

# gpu4, gpu4_2
- gpu4
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain_stream_event.py

$ NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain.py

- gpu4_2
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain_stream_event.py

$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
$ NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain.py
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook, bf16_compress_wrapper
# from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
from PowerSGD_hook_stream_event import PowerSGDState, powerSGD_hook
from transformers import AutoConfig

from dataset import Task
from model import Z2allConfig, Z2allForCausalLM
from transformers import AutoTokenizer
import utils
from utils import get_logger, estimate_mfu, configure_optimizers, ResLog, save_run_exp_config


os.environ["NCCL_IB_DISABLE"] = "1"  # disable infiniband
os.environ["NCCL_IBEXT_DISABLE"] = "1"  # 有用啦!
os.environ["NCCL_P2P_DISABLE"] = "1"  # disable p2p
os.environ["OMP_NUM_THREADS"] = "8"  # set the number of threads for OpenMP
# os.environ["NCCL_DEBUG"] = "WARN"  # set NCCL debug level, ["WARN", "INFO"]，用于测试

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"  # 设置每个 CUDA 设备的最大并行内核执行数，速度还快了？


# 用于torch.compile，需要PyTorch>=2.0
if torch.__version__ >= "2.0.0":
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128  # 原来是64，有警告，设大点加快编译

# 用于找到pad的id
# tokenizer = AutoTokenizer.from_pretrained("tokenizer/hf_bbpe_tokenizer", trust_remote_code=True)

# -----------------------------------------------------------------------------
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
# I/O
out_dir = "out"
out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
eval_interval = 200  # 每eval_interval个step验证一次
log_interval = 1
eval_iters = 3  # 每次验证的step数
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
resume = False  # if True, resume training from the last checkpoint
# wandb logging
use_reslog = True  # wandb用起来有问题，改为自己的日志和画图工具
reslog_dir = "reslog"
reslog_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
reslog_save_interval = 1  # 想快速看结果，用小点的数
# data
train_bin_dir = "data/02_train_data_more/01_bin_for_train_hf"
valid_bin_dir = "data/02_train_data_more/02_bin_for_valid_hf"
num_workers = 0  # 数据加载器的工作进程数
## global_batch_size=batch_size*gradient_accumulation_steps*ddp_world_size
batch_size = 8  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
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
    save_run_exp_config(exp_config, os.path.join(out_dir, "exp_config.py"))

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
# 训练所需，梯度缩放器，优化器，学习率调度器，测试函数
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type, logger, master_process)
if resume and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory
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
# 使用DDP包裹模型
# wrap model into DDP container, 只有在从头开始训练时才会进行包裹
if ddp:
    # model.bfloat16()
    _ = logger.info(f"wrapping model into DDP container") if master_process else None
    model = DDP(model, device_ids=[ddp_local_rank],
                gradient_as_bucket_view=True, static_graph=False)  # 有梯度累加时，static_graph=True有问题?

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
if ddp:
    warm_for_bucket_rebuilt_time = time.time()
    for _ in range(2):
        with ctx:
            model_outputs = model(X, Y)
            loss = model_outputs["loss"]
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        if utils.FusedAdam is not None:
            optimizer.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
    torch.distributed.barrier()
    _ = logger.info(f"warm_for_bucket_rebuilt_time: {time.time() - warm_for_bucket_rebuilt_time:.4f}s") if master_process else None

# -----------------------------------------------------------------------------
# 预热后，设置ddp com hook，还是会取world_size的平均值
def stream_wrapper(hook):
    def wrapper(state, bucket):
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            event.wait(s)
            return hook(state, bucket)
    return wrapper
if ddp:
    if use_powerSGD_hook:
        # 若没有resume，则初始化PowerSGDState
        if powerSGD_state is None:
            powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
                                warm_start=True, use_error_feedback=True, start_powerSGD_iter=2, 
                                min_compression_rate=0.5, orthogonalization_epsilon=1e-6)
        if use_bf16_compress_hook:
            model.register_comm_hook(powerSGD_state, bf16_compress_wrapper(powerSGD_hook))
        else:
            # model.register_comm_hook(powerSGD_state, powerSGD_hook)
            model.register_comm_hook(powerSGD_state, stream_wrapper(powerSGD_hook))
    elif use_bf16_compress_hook:
        model.register_comm_hook(process_group, bf16_compress_hook)

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
    
    # 前向传播和反向传播，梯度更新
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1  # 只在最后的forawrd-backward进行同步操作
        with ctx:
            model_outputs = model(X, Y)
            loss = model_outputs["loss"]
            loss = loss / gradient_accumulation_steps
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = next(train_batch_iter)
        # 最后一次backward和通信所耗费时间
        backward_comm_time = time.time()
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()  # 同步的时候会自动进行梯度的all-reduce，并且取所有的word_size的平均值
    if ddp:
        torch.distributed.barrier()  # 只在测试通信耗时的时候用
    backward_comm_time = time.time() - backward_comm_time
    
    torch.cuda.synchronize()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)  # 若没有unscale，进行unscale并且更新
    scaler.update()  # 更新scaler的缩放因子
    # flush the gradients as soon as we can, no need for this memory anymore
    if utils.FusedAdam is not None:
        optimizer.zero_grad()  # apex fused adamw上已经设置了set_to_none了
    else:
        optimizer.zero_grad(set_to_none=True)  # pytorch的需要在这设置为None，清空显存
        
    # 输出结果
    # timing and logging
    train_time1 = time.time()
    dt = train_time1 - train_time0
    train_time0 = train_time1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        # 调用.item()方法会导致CPU等待GPU计算完成，因为需要将数据从GPU内存复制到CPU内存。
        lossf = loss.item() * gradient_accumulation_steps
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
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt:.4f}s | mfu {running_mfu*100:.2f}% | backward_comm: {backward_comm_time:.4f}s"
        )
    iter_num += 1
    local_iter_num += 1

    # 中止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # gloo退出有问题，这行代码不会退出
