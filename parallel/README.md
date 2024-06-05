# 分布式训练

代码见：`llm-zero2all/parallel`

## 01 分布式数据并行（Distributed Data Parallel）

### 1 使用pytorch进行DDP训练

#### 多节点训练
我的训练环境：局域网内，多个节点代表多个docker镜像，其中主节点宿主机的地址：10.10.24.107，主节点docker与宿主机的端口映射是9527:30846

因为节点是docker镜像，所以主节点设置master时，需要设置docker镜像内的地址和端口（如localhost:9527）；而其他节点设置master时，需要设置主节点宿主机的地址和端口（如10.10.24.107:30846）

此时，使用pytorch进行多节点训练时就会出现问题了，通常使用torchrun运行，如命令：

```bash
# 主节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
# 其他节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
```

这个命令会卡住，即表示无法正常通信，因为局域网内主节点下的docker镜像可能不知道10.10.24.107:30846表示自己，所以只能用localhost:9527来表示自己，如命令：

```bash
# 主节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 pretrain.py
# 其他节点
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 pretrain.py
```

这个命令虽然使得节点之间可以通信了，但是进行第一次连接后开始训练时又会出现问题，因为torchrun的原理是在主节点上管理group内的信息，当其他节点会将主节点的master信息更新为自己的信息，但是主节点的master地址是localhost:9527，所以导致后续其他节点无法访问到主节点了。。。

这里我的解决方法比较粗暴，我直接修改代码torch/distributed/elastic/agent/server/api.py中的_get_master_addr_port来获取真正的master地址（只在其他节点下的torch环境下修改），如下，直接设置为主节点宿主机的地址，后面再次运行命令即可正常通信训练了：

```python
@staticmethod
def _get_master_addr_port(store: Store) -> Tuple[str, int]:
    # master_addr = store.get("MASTER_ADDR").decode(encoding="UTF-8")
    # master_port = int(store.get("MASTER_PORT").decode(encoding="UTF-8"))
    master_addr = "10.10.24.107"
    master_port = "30846"
    return (master_addr, master_port)
```

#### gpu4，训练超参数设置

训练命令：`OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py`

tokens per iteration will be: 1,048,576

breaks down as: 16 grad accum steps * 4 processes * 8 batch size * 2048 max seq len

性能：14.71s, 49.33%, 48.08GB

---

训练命令：`OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py`

tokens per iteration will be: 1,048,576

breaks down as: **8 grad accum steps** * 4 processes * **16 batch size** * 2048 max seq len

性能：14.30s, 50.73%, 70.97GB

---

训练命令：`OMP_NUM_THREADS=8 torchrun --standalone --nproc_per_node=4 pretrain.py`

tokens per iteration will be: 1,048,576

breaks down as: 8 grad accum steps * 4 processes * 16 batch size * 2048 max seq len

**设置 num_workers 从 2 -> 0**

性能：13.77s, 52.69%, 70.97GB

---

#### gpu4, gpu4_2
tokens per iteration will be: 1,048,576

breaks down as: 4 grad accum steps * 8 processes * 16 batch size * 2048 max seq len

计算7s，通信19s

#### 注意

1. 使用torchrun时，因为我的解释器路径太长而被截断了。。。
可以创建一个符号链接指向原python解释器，缩短路径长度，注意将路径内容改为自己的：

    `ln -s /path/to/raw/python /path/to/link`

    然后将torchrun脚本第一行从`#!/path/to/raw/python`改为`#!/path/to/link`

2. 使用nccl通信后端，代码在DDP()包裹模型时会卡住，通过禁用P2P通信解决，
见：https://github.com/pytorch/pytorch/issues/23074，但是会影响通信效率。或者可以改为gloo通信后端。

    GPU通信方式：https://zhuanlan.zhihu.com/p/74217534

### 2 梯度通信优化

使用DDP的`torch.distributed.algorithms.ddp_comm_hooks`

1. `bf16_compress_hook`，压缩到bf16，allreduce通信，解压缩回来

    ```python
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook

    process_group = init_process_group(backend=ddp_backend)

    ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    ```

2. `PowerSGDState, powerSGD_hook`，使用PowerSGD算法压缩梯度，压缩率更高，但可能误差大点；参数细节见源码。

    ```python
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook

    process_group = init_process_group(backend=ddp_backend)

    state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
                          warm_start=True, use_error_feedback=True, start_powerSGD_iter=2, 
                          min_compression_rate=0.5, orthogonalization_epsilon=1e-6)
    model.register_comm_hook(state, powerSGD_hook)  # 会取平均
    # 或者两个通信优化叠加起来
    # model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))  # 会取平均
    ```

    注意PowerSGDState的状态需要保存，用于resume

参考：

* https://medium.com/pytorch/accelerating-pytorch-ddp-by-10x-with-powersgd-585aef12881d
* https://github.com/epfml/powersgd
* https://pytorch.org/docs/stable/ddp_comm_hooks.html

**注意**，

要使用nccl后端通信
1. gloo，不支持bfloat16，使用PowerSGD时会卡住，强行退出时GPU不会立即释放
2. nccl，需要设置NCCL_IB_DISABLE=1，NCCL_IBEXT_DISABLE=1，NCCL_P2P_DISABLE=1

使用nccl时，如果没有nvlink，则需要设置NCCL_P2P_DISABLE=1。没有nvlink时，在单节点下DP比DDP更快，但是DP不支持多节点训练。

最终我使用nccl后端，并且设置NCCL_IB_DISABLE=1，NCCL_IBEXT_DISABLE=1，NCCL_P2P_DISABLE=1，见：https://github.com/microsoft/DeepSpeedExamples/issues/542

#### 使用cuda的stream和event使得PowerSGD的计算和通信进一步重叠

原先PowerSGD通信时，backward没有计算，因为PowerSGD中还有计算kernel，会阻塞backward中的计算kernel。

所以我对PowerSGD基础上添加一个Wrapper，将通信放在另一个stream中，通过event实现同步。

```python
def stream_wrapper(hook):
    def wrapper(state, bucket):
        event = torch.cuda.Event(enable_timing=False)
        event.record(torch.cuda.current_stream())
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            event.wait(s)
            return hook(state, bucket)
    return wrapper
powerSGD_state = PowerSGDState(process_group=process_group, matrix_approximation_rank=32,
                               warm_start=True, use_error_feedback=True, start_powerSGD_iter=2, 
                               min_compression_rate=0.5, orthogonalization_epsilon=1e-6)
model.register_comm_hook(powerSGD_state, stream_wrapper(powerSGD_hook))
```

注意，optimizer.step()前可能需要进行同步`torch.cuda.synchronize()`，防止梯度还没通信完

修改见：pretrain_stream_event.py和PowerSGD_hook_stream_event.py

**性能变化**，f32_PowerSGD，一个iter，2M的tokens下：

前：17.2556s | mfu 53.16% | backward_comm: 3.1694s

后：16.9890s | mfu 54.00% | backward_comm: 2.6669s

### 3 我的DDP实现

基于Megatron-LM源码，简化了实现代码，添加了部分其他特性。

1. 使用hook，如下，当模型参数的梯度进行累加时触发

```python
# Expand so we get access to grad_fn.
param_tmp = param.expand_as(param)
# Get the gradient accumulator function.
grad_acc = param_tmp.grad_fn.next_functions[0][0]
grad_acc.register_hook(...)
```

在 PyTorch 中，next_functions 是一个属性，存在于 Function 对象中。Function 对象是 PyTorch 自动微分系统的一部分，用于表示创建 Tensor 的操作。每个 Tensor 都有一个 grad_fn 属性，指向创建它的 Function。

next_functions 是一个元组列表，每个元组包含两个元素：一个 Function 对象和一个整数。Function 对象表示参与到创建当前 Tensor 的操作，整数表示这个操作在其输出中的索引。

例如，假设我们有以下的代码：

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = x * y
```

在这个例子中，z 是通过乘法操作创建的，所以它的 grad_fn 是一个 MulBackward0 对象。这个对象的 next_functions 属性包含两个元素，分别对应 x 和 y。每个元素是一个元组，包含一个 AccumulateGrad 对象（表示梯度累积操作，也表示这是一个叶子节点）和一个整数（表示这个操作在其输出中的索引）。

所以，next_functions 实际上是一个记录了创建当前 Tensor 的操作的历史记录。这个历史记录在反向传播过程中被用来计算梯度。

## 02 分布式优化器（Distributed Optimizer, ZeRO1）

### 实现overlap_optim_step

1. `optimizer.bucket_to_optim_param_groups_map`, 用于optimizer.step()

2. `optimizer.bucket_to_model_params_map`, `optimizer.bucket_to_main_params_map`, 两个目的：

    * `main_param.grad = model_param.main_grad.float()`，复制梯度，用于optim参数更新
    * `model_param.copy_(main_param)`，将更新后的optim参数复制回model

3. 注意，DistributedOptimizer下是

    * `optimizer.bucket_to_model_params_map`，用于复制梯度，因为main_grad（或grad）是跟着model_param的，不能直接取shard_model_param的main_grad
    * `optimizer.bucket_to_shard_model_params_map`，用于复制参数回模型
    * `optimizer.bucket_to_shard_main_params_map`

## 03 张量并行（Tensor Parallel）

### 计算和通信重叠

张量并行中的计算和通信重叠需要考虑下，主要是`ColumnParallelLinear`反向传播中的2个计算和2个通信的重叠，使用的是序列并行版本的张量并行。

见：megatron\core\tensor_parallel\layers.py的`LinearWithGradAccumulationAndAsyncCommunication`

**重叠方法**，计算和通信重叠，使用async_op=True（表示后续kernel操作与该通信没有依赖关系，因为没有依赖关系，所以可以被调度到不同的硬件队列中，这样就算在同一个流中也可以进行并行），然后中间加一个handle.wait()来同步

1, 2需要(input, grad_output, weight), 输出(**total_input**, **grad_input**)

4, 5需要(**grad_input**, grad_output, **total_input**)

1. 通信: handle = all_gather(**total_input**, input, group=..., **async_op=True**)
2. 计算: **grad_input** = grad_output.matmul(weight)
3. handle.wait()
4. 通信: handle = reduce_scatter(sub_grad_input, **grad_input**, group=..., **async_op=True**)
5. 计算: grad_weight = grad_output.t().matmul(**total_input**), grad_bias = ...
6. handle.wait()
7. return sub_grad_input, grad_weight, grad_bias, ...

### 为什么要设置CUDA_DEVICE_MAX_CONNECTIONS=1

Megatron-LM中有多个注释说需要设置`CUDA_DEVICE_MAX_CONNECTIONS=1`

```python
# Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
# gather is scheduled before the input gradient computation
```

其他人的解释：https://github.com/NVIDIA/Megatron-LM/issues/533

#### cuda流和硬件队列

这个问题涉及到cuda中的流和硬件队列，与kernel并行有关。

简单来说，`CUDA_DEVICE_MAX_CONNECTIONS=1`会强制设置cuda硬件队列（kernel执行队列）为1，即将 GPU 上的kernel执行顺序强制为来自主机的kernel排队顺序。

对于 GEMM 和 TP 通信重叠，它允许在 GEMM 之前调度 GPU 中的通信kernel，以便在 GEMM 占用所有 GPU 资源之前让通信内核分配 GPU 资源（即防止计算kernel占用所有资源，从而导致通信kernel无法开始调度，可能是通信kernel占用的资源少，所以需要先调度，保证计算和通信重叠）。本质上就像只有一个cuda流。

参考资料：

1. [CUDA的kernel并行](https://blog.csdn.net/feng__shuai/article/details/122441011)

2. [CUDA流：利用并行执行提高性能](https://blog.csdn.net/Algabeno/article/details/129152135)

#### copilot提问

**问**

代码顺序为：

handle = torch.distributed._all_gather_base( all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True)

grad_input = grad_output.matmul(weight) 

有两个硬件队列，都为空闲，那么这两个kernel的执行顺序是不是不一定是顺序的，然后可能两个kernel是同时调度的，分配资源，此时如果计算kernel耗费的资源过多，会不会导致通信kernel并没有立即启动，而是等待资源释放

而如果硬件队列只有一个，那么是不是不会发生这种情况，因为通信kernel会先进行资源的分配，如果还有剩余则会开启计算kernel

**答**

是的，你的理解是正确的。在多队列环境中，两个内核可能会同时调度，这取决于调度器如何管理这些队列。如果计算内核消耗的资源过多，可能会导致通信内核等待资源释放才能启动。

然而，如果只有一个硬件队列，那么这种情况就不太可能发生，因为通信内核会先进行资源的分配，如果还有剩余资源，计算内核才会启动。这是因为在一个队列中，任务通常按照它们被提交的顺序执行。

但是，这也取决于具体的硬件和驱动程序实现，以及你的代码如何使用异步操作。在某些情况下，即使在单队列中，如果使用了异步操作，也可能会出现资源竞争的情况。

## 04 流水线并行（Pipeline Parallel）



## 05 上下文并行（Context Parallel）

注意，有context时，mpu.get_data_parallel_group()返回的是dp-cp组，因为dp和cp都需要进行grad的all-reduce

Megatron-LM中有context parallel时，为loss设置scale时不太懂（即取所有token和microbatch的平均）

见：megatron\core\pipeline_parallel\schedules.py的`output_tensor /= num_microbatches`和pretrain_gpt.py的`loss[0] * args.context_parallel_size`

Context Parallel学习资料：

* https://www.zhihu.com/question/637961859/answer/3397275232
* https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html