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

# gpu4
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py
# test
$ OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=gloo  # gloo
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain.py --ddp_backend=nccl  # nccl
$ OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 NCCL_BUFFLE_SIZE=16777216 torchrun --standalone --nproc_per_node=4 pretrain.py
$ OMP_NUM_THREADS=8 NCCL_BUFFLE_SIZE=16777216 NCCL_P2P_LEVEL=5 torchrun --standalone --nproc_per_node=4 pretrain.py # error

# gpu4, gpu4_2
- gpu4
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
$ NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
- gpu4_2
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain.py
$ OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
$ NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain.py
"""

import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial

import torch
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

from transformers import AutoConfig

from dataset import Task
from model import Z2allConfig, Z2allForCausalLM
from transformers import AutoTokenizer
import utils
from utils import get_logger, estimate_mfu, configure_optimizers, ResLog


# 用于找到pad的id
# tokenizer = AutoTokenizer.from_pretrained("tokenizer/hf_bbpe_tokenizer", trust_remote_code=True)

# 见：https://huggingface.co/docs/transformers/perf_train_gpu_many
# 使用nccl时，如果没有nvlink，则需要设置NCCL_P2P_DISABLE=1
# 没有nvlink时，在单节点下DP比DDP更快，但是DP不支持多节点训练
# 因为我的环境没有nvlink，所以我使用的是gloo后端
ddp_backend = "nccl"  # ddp backend, can be 'nccl', 'gloo'

# 用于torch.compile，需要PyTorch>=2.0
if torch.__version__ >= "2.0.0":
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128  # 原来是64，有警告，设大点加快编译

# -----------------------------------------------------------------------------
# 实验设置
# I/O
out_dir = "out"
out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
eval_interval = 100  # 每eval_interval:100个step验证一次
log_interval = 1
eval_iters = 100  # 每次验证的step:100数
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume'
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
batch_size = 16  # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 2048
# model
vocab_size = 64320  # 实际是64012个，写大点方便扩展，注意最好是8的倍数，见指导：https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html#tc-guidelines-padding
hidden_dim = 2048
intermediate_size = 5632
n_layers = 24
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
attention_dropout = 0.0  # TODO: 或许不用设置dropout
dropout1 = 0.0
dropout2 = 0.0
residual_in_fp32 = True  # 残差连接是否使用fp32
# adamw optimizer
## gradient_accumulation_steps=gradient_accumulation_steps*ddp_world_size
gradient_accumulation_steps = 128  # used to simulate larger batch sizes
learning_rate = 4e-4  # max learning rate
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
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
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
    process_group = init_process_group(backend=ddp_backend)
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
    # wandb.init(project=wandb_project, name=wandb_run_name, config=config)
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
# 模型初始化，配置初始化
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
# model init
if init_from == "scratch":
    # init a new model from scratch
    _ = logger.info("Initializing a new model from scratch") if master_process else None  # 通过这种方式可以避免在非master进程中打印
    model_config = Z2allConfig(**config)
    model = Z2allForCausalLM(model_config)
elif init_from == "resume":  # TODO
    _ = logger.info(f"Resuming training from {out_dir}") if master_process else None
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_config = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    
    # create the model
    model = Z2allForCausalLM(model_config)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    # 使用了torch.compile才会出现这个问题
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)  # 修改键名
    model.load_state_dict(state_dict)
    
    reslog.load(os.path.join(out_dir, "reslog.pkl"))  # 读取中止的实验日志
    
    # 训练时候的信息
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
model.to(device)

# -----------------------------------------------------------------------------
# 训练所需，梯度缩放器，优化器，学习率调度器
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# optimizer
optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type, logger, master_process)
if init_from == "resume" and "optimizer" in checkpoint:
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
# wrap模型
# 编译模型，需要PyTorch>=2.0
if compile and torch.__version__ >= "2.0":
    if master_process:
        logger.info("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
# 使用DDP包裹模型
# wrap model into DDP container
if ddp:
    # model.bfloat16()
    _ = logger.info(f"wrapping model into DDP container") if master_process else None
    model = DDP(model, device_ids=[ddp_local_rank], gradient_as_bucket_view=True, static_graph=True)
    # model = DDP(model, device_ids=[ddp_local_rank], gradient_as_bucket_view=True)

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
    
# model.register_comm_hook(process_group, bf16_compress_hook)
def noop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    fut = torch.futures.Future()
    fut.set_result(bucket.buffer())
    return fut

def test_bfloat16(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    fut = torch.futures.Future()
    
    temp = bucket.buffer().bfloat16()
    dist.all_reduce(temp)
    
    fut.set_result(temp.float())
    
    return fut

# -----------------------------------------------------------------------------
# 开始训练
train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)  # fetch the very first batch
t0 = time.time()
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

    # 测试
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 110 and master_process:
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
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                logger.info(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_config.save_pretrained(out_dir)  # 单独保存模型配置
                reslog.save(os.path.join(out_dir, "reslog.pkl"))
                # model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)  # TODO
    if iter_num == 0 and eval_only:
        break
    
    # model.register_comm_hook(state=None, hook=noop)
    # model.register_comm_hook(state=None, hook=test_bfloat16)
    # model.register_comm_hook(process_group, bf16_compress_hook)
    
    state = PowerSGDState(process_group=process_group, matrix_approximation_rank=128,
                          start_powerSGD_iter=2, min_compression_rate=0.5)
    model.register_comm_hook(state, powerSGD_hook)  # 会取平均
    
    # python pretrain.py --batch_size=16 --gradient_accumulation_steps=16
    # OMP_NUM_THREADS=8 NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=4 pretrain.py
    if master_process:
        print("grad为float32测试")
    # grad是float32
    # ctx = nullcontext()
    with ctx:
        # 先运行5个，防止速度有差别
        for i in range(1000):
            # 同步耗费时间，同步包含计算和通信
            model.zero_grad(set_to_none=True)
            test0 = time.time()
            model.require_backward_grad_sync = True
            # model.zero_grad(set_to_none=True)
            model_outputs = model(X, Y)
            loss = model_outputs["loss"]
            with torch.autocast(enabled=False, device_type=device_type):
                loss.backward()
            test1 = time.time()
            if master_process:
                print(f"{i} 时间: {test1 - test0:.4f}s")
                get0 = time.time()
                print("grad: ", model.module.model.layers[0].self_attn.o_proj.weight.grad[0, 0])
                get1 = time.time()
                print(f"get time: {get1 - get0:.4f}s")
        sync0 = time.time()
        torch.cuda.synchronize()
        sync1 = time.time()
        if master_process:
            print(f"sync time: {sync1 - sync0:.4f}s")
                
        # 同步耗费时间，同步包含计算和通信
        model.zero_grad(set_to_none=True)
        test00 = time.time()
        model.require_backward_grad_sync = True
        # model.zero_grad(set_to_none=True)
        model_outputs = model(X, Y)
        loss = model_outputs["loss"]
        with torch.autocast(enabled=False, device_type=device_type):
            loss.backward()
        test10 = time.time()
        if master_process:
            print(f"同步耗费时间: {test10 - test00:.4f}s")
            get0 = time.time()
            print("grad: ", model.module.model.layers[0].self_attn.o_proj.weight.grad[0, 0])
            get1 = time.time()
            print(f"get time: {get1 - get0:.4f}s")
        sync0 = time.time()
        torch.cuda.synchronize()
        sync1 = time.time()
        if master_process:
            print(f"sync time: {sync1 - sync0:.4f}s")
            
        # 异步耗费时间，异步包含计算
        model.zero_grad(set_to_none=True)
        test01 = time.time()
        model.require_backward_grad_sync = False
        # model.zero_grad(set_to_none=True)
        model_outputs = model(X, Y)
        loss = model_outputs["loss"]
        with torch.autocast(enabled=False, device_type=device_type):
            loss.backward()
        test11 = time.time()
        if master_process:
            print(f"异步耗费时间: {test11 - test01:.4f}s")
            get0 = time.time()
            print("grad: ", model.module.model.layers[0].self_attn.o_proj.weight.grad[0, 0])
            get1 = time.time()
            print(f"get time: {get1 - get0:.4f}s")
        sync0 = time.time()
        torch.cuda.synchronize()
        sync1 = time.time()
        if master_process:
            print(f"sync time: {sync1 - sync0:.4f}s")
                
        # 同步耗费时间，同步包含计算和通信
        model.zero_grad(set_to_none=True)
        test02 = time.time()
        model.require_backward_grad_sync = True
        # model.zero_grad(set_to_none=True)
        model_outputs = model(X, Y)
        loss = model_outputs["loss"]
        with torch.autocast(enabled=False, device_type=device_type):
            loss.backward()
        test12 = time.time()
        if master_process:
            print(f"同步耗费时间: {test12 - test02:.4f}s")
            get0 = time.time()
            print("grad: ", model.module.model.layers[0].self_attn.o_proj.weight.grad[0, 0])
            get1 = time.time()
            print(f"get time: {get1 - get0:.4f}s")
        sync0 = time.time()
        torch.cuda.synchronize()
        sync1 = time.time()
        if master_process:
            print(f"sync time: {sync1 - sync0:.4f}s")

    # if master_process:
    #     print("grad为bfloat16测试")
    # # model.bfloat16()
    # # 先运行5个，防止速度有差别
    # for i in range(5):
    #     # 同步耗费时间，同步包含计算和通信
    #     test0 = time.time()
    #     model.require_backward_grad_sync = True
    #     model_outputs = model(X, Y)
    #     loss = model_outputs["loss"]
    #     loss.backward()
    #     test1 = time.time()
    #     if master_process:
    #         print(f"{i} 时间: {test1 - test0:.4f}s")
            
    # # 同步耗费时间，同步包含计算和通信
    # test0 = time.time()
    # model.require_backward_grad_sync = True
    # model_outputs = model(X, Y)
    # loss = model_outputs["loss"]
    # loss.backward()
    # test1 = time.time()
    # if master_process:
    #     print(f"同步耗费时间: {test1 - test0:.4f}s")
        
    # # 异步耗费时间，异步包含计算
    # test0 = time.time()
    # model.require_backward_grad_sync = False
    # model_outputs = model(X, Y)
    # loss = model_outputs["loss"]
    # loss.backward()
    # test1 = time.time()
    # if master_process:
    #     print(f"异步耗费时间: {test1 - test0:.4f}s")
            
    # # 同步耗费时间，同步包含计算和通信
    # test0 = time.time()
    # model.require_backward_grad_sync = True
    # model_outputs = model(X, Y)
    # loss = model_outputs["loss"]
    # loss.backward()
    # test1 = time.time()
    # if master_process:
    #     print(f"同步耗费时间: {test1 - test0:.4f}s")
    
    break
    
    # torch.cuda.synchronize()
    # test2 = time.time()
    # if master_process:
    #     print(f"同步耗费时间: {test2 - test1:.4f}s")
    
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
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()  # 同步的时候会自动进行梯度的all-reduce，并且取所有的word_size的平均值
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
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
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
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt:.4f}s | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # 中止条件
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()  # gloo退出有问题，这行代码不会退出
