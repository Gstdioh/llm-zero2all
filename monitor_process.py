"""
监控进程

#* 表明了需要修改的地方

可以只在rank0上运行，其会自动使用ssh发送运行命令到其他节点上（需要在代码中设置远程服务器的地址密码等信息）

当进程异常中断时，会自动重新启动进程（进行resume命令）
"""

import subprocess
import time
import os
from datetime import datetime
import argparse

import paramiko


parser = argparse.ArgumentParser()

parser.add_argument("--out_dir", type=str, default=None, help="输出目录")

parser.add_argument("--remote", action="store_true", default=False, help="是否有remote")
parser.add_argument("--hostname", type=str, default="123.456.123.456", help="remote的hostname")
parser.add_argument("--port", type=int, default=9527, help="remote的port")
parser.add_argument("--username", type=str, default="root", help="remote的username")
parser.add_argument("--password", type=str, default="123456", help="remote的password")
parser.add_argument("--remote_torchrun", type=str, default="abspath_to_torchrun", help="remote的torchrun的绝对路径")
parser.add_argument("--remote_workspace", type=str, default="abspath_to_remote_workspace", help="remote的工作目录的绝对路径")

parser.add_argument("--rank0_start_command", required=True, type=str, default="abspath_to_torchrun", help="rank0执行的命令")
parser.add_argument("--remote_start_command", type=str, default="abspath_to_remote_workspace", help="remote执行的命令")

args = parser.parse_args()

# -----------------------------------------------------------------------------
# 提前设置好输出目录
out_dir = "out"
out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
if args.out_dir is not None:
    out_dir = args.out_dir

# -----------------------------------------------------------------------------
# rank0需要执行的命令，环境变量只能额外设置
# 获取当前的环境变量
env = os.environ.copy()
# 添加新的环境变量
env["OMP_NUM_THREADS"] = "8"
# rank0要执行的命令
rank0_start_command = f"{args.rank0_start_command} --out_dir={out_dir}"
rank0_resume_command = rank0_start_command + " --resume"
# 需要转换为列表的形式
rank0_start_command = rank0_start_command.split(" ")
rank0_resume_command = rank0_resume_command.split(" ")
# -----------------------------------------------------------------------------
# 远程服务器需要执行的命令，注意需要切换到对应的目录，激活对应的conda环境，然后才能执行命令
# 远程服务器的命令，预先设好命令的绝对路径
if args.remote:
    remote_torchrun = args.remote_torchrun
    remote_workspace = args.remote_workspace
    # 将remote_start_command中的torchrun替换为remote_torchrun
    remote_start_command = args.remote_start_command.split(" ")
    for i in range(len(remote_start_command)):
        if remote_start_command[i] == "torchrun":
            remote_start_command[i] = remote_torchrun
            break
    remote_start_command = " ".join(remote_start_command)

    # 构建远程服务器需要运行的命令
    remote_start_command = f"cd {remote_workspace} ; {remote_start_command} --out_dir={out_dir}"
    remote_resume_command = remote_start_command + " --resume"

# -----------------------------------------------------------------------------
# 如果输出目录中存在best1_reslog.pkl或者best2_reslog.pkl文件，说明已经训练过了，可以直接resume，否则从头开始
if os.path.exists(os.path.join(out_dir, "best1_reslog.pkl")) or os.path.exists(os.path.join(out_dir, "best2_reslog.pkl")):
    rank0_command = rank0_resume_command
    remote_command = remote_resume_command
else:
    rank0_command = rank0_start_command
    remote_command = remote_start_command
    
# -----------------------------------------------------------------------------
# 创建一个新的进程来执行命令
p = subprocess.Popen(rank0_command, env=env)
time.sleep(1)  # 确保rank0进程已经启动
# -----------------------------------------------------------------------------
# 执行远程服务器的命令
# 首先创建一个SSH客户端
if args.remote:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 连接到远程服务器
    ssh.connect(hostname=args.hostname, port=args.port, username=args.username, password=args.password)
    # 执行远程服务器的命令
    remote_stdin, remote_stdout, remote_stderr = ssh.exec_command(remote_command)

# -----------------------------------------------------------------------------
# monitor基本设置
monitor_basename = os.path.basename(__file__)
monitor_prefix = f"[{monitor_basename}]: "
wait_time = 30  # 重启进程的等待时间

# -----------------------------------------------------------------------------
# 开始监控进程
while True:
    try:
        # 检查进程是否已经结束
        if p.poll() is not None:
            time.sleep(5)  # 确保下面的输出不会和其他进程的输出混在一起
            
            # -----------------------------------------------------------------------------
            # 等待一段时间再重新启动，期间可以按下Ctrl+C中断
            print(monitor_prefix + f"启动的进程结束, 返回: code {p.returncode}, {wait_time}秒后重新启动, 可以按下两次Ctrl+C中断")
            time.sleep(wait_time)
            
            print(monitor_prefix + f"重新启动进程: {' '.join(rank0_command)}")
            
            # 如果输出目录中存在best1_reslog.pkl或者best2_reslog.pkl文件，说明已经训练过了，可以直接resume，否则从头开始
            if os.path.exists(os.path.join(out_dir, "best1_reslog.pkl")) or os.path.exists(os.path.join(out_dir, "best2_reslog.pkl")):
                rank0_command = rank0_resume_command
                remote_command = remote_resume_command
            else:
                rank0_command = rank0_start_command
                remote_command = remote_start_command
            
            # 重新启动rank0进程
            p = subprocess.Popen(rank0_command, env=env)
            time.sleep(1)  # 确保rank0进程已经启动
            # 重新启动远程服务器的命令
            if args.remote:
                remote_stdin, remote_stdout, remote_stderr = ssh.exec_command(remote_command)
        else:
            # 等待一段时间再次检查
            time.sleep(2)
    except:
        time.sleep(5)  # 确保下面的输出不会和其他进程的输出混在一起
        
        print(monitor_prefix + "启动的进程异常中断")
        p.terminate()
            
        # -----------------------------------------------------------------------------
        # 等待一段时间再重新启动，期间可以按下Ctrl+C中断
        print(monitor_prefix + f"启动的进程结束, 返回: code {p.returncode}, {wait_time}秒后重新启动, 可以按下两次Ctrl+C中断")
        time.sleep(wait_time)
        
        print(monitor_prefix + f"重新启动进程: {' '.join(rank0_command)}")
        
        # 如果输出目录中存在best1_reslog.pkl或者best2_reslog.pkl文件，说明已经训练过了，可以直接resume，否则从头开始
        if os.path.exists(os.path.join(out_dir, "best1_reslog.pkl")) or os.path.exists(os.path.join(out_dir, "best2_reslog.pkl")):
            rank0_command = rank0_resume_command
            remote_command = remote_resume_command
        else:
            rank0_command = rank0_start_command
            remote_command = remote_start_command
        
        # 重新启动rank0进程
        p = subprocess.Popen(rank0_command, env=env)
        time.sleep(1)  # 确保rank0进程已经启动
        # 重新启动远程服务器的命令
        if args.remote:
            remote_stdin, remote_stdout, remote_stderr = ssh.exec_command(remote_command)
