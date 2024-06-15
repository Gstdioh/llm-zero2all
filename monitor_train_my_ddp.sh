#!/bin/bash

# 将标准输出和标准错误同时也输出到文件中
# ./local.sh 2>&1 | tee monitor_out.txt

REMOTE1_ARGS=(
    --remote_torchrun "abspath_to_torchrun"
    --remote_workspace "abspath_to_remote_workspace"
    --hostname "123.456.123.456"
    --port 9527
    --username "root"
    --password "123456"
)

# 前面不加环境变量，可以在输入命令的时候加，或者在monitor_process.py中加
RNK0_CMD="torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=9527 train_my_ddp.py"

# 这里需要加环境变量，应为是通过ssh发出的命令
REMOTE_CMD="OMP_NUM_THREADS=8 torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=10.10.24.107 --master_port=30846 train_my_ddp.py"

python monitor_process.py --remote \
    ${REMOTE1_ARGS[@]} \
    --rank0_start_command "${RNK0_CMD}" \
    --remote_start_command "${REMOTE_CMD}"
