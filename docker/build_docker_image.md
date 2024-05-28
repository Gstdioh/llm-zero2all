# build docker image

构建docker镜像用于该项目

环境：pytorch2.0.1, cuda11.4, flash-attn2.1.0, rotary-emb0.1, xentropy-cuda-lib0.1, apex0.1, xformers0.0.24等

**注意**，xformers的bug在镜像中处理过了，处理方法见：[xformers的bug处理方法](../model/README.md/#6-swiglu-xformers)

## 构建过程

我从基本的ubuntu20.04开始构建，以下是构建该docker镜像的详细过程

```bash
#$ in your system
docker pull ubuntu:20.04
docker run -it --name llm_2.0.1_11.4 --gpus all -p 9530:22 -v /home/guoliuyang/code:/code your_image_id bash

#$ in docker
apt-get update
apt-get install vim
apt-get install wget

cd ~
mkdir software
cd software

# install cuda11.4
# cat /etc/os-release, show your system
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
chmod 777 ./cuda_11.4.4_470.82.01_linux.run
apt-get install libxml2  # set -> 6 -> 70
apt-get install build-essential  # install gcc
./cuda_11.4.4_470.82.01_linux.run
# in CUDA Installer, you can unset Driver
# ~
# set cuda11.4 path
vim ~/.bashrc
# ---------- in .bashrc, set cuda11.4 path:
export PATH=/usr/local/cuda-11.4/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64
# ---------- wq
source ~/.bashrc
# check if the installation was successful:
nvcc -V  # you will see the cuda11.4

# conda, install miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 777 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# when install, you can set path into /root/software/miniconda3/bin
# ~
# set conda path
vim ~/.bashrc
# ---------- in .bashrc, set conda path:
export PATH=/root/software/miniconda3/bin:$PATH
# ---------- wq
source ~/.bashrc
conda init
source ~/.bashrc

# create env LLM
conda create -n LLM python=3.10
vim ~/.bashrc
# ---------- in .bashrc, set auto into your env:
conda activate LLM
# ---------- wq
source ~/.bashrc

# in LLM, install pytorch2.0.1-11.7
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 设置清华镜像，加速下载
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# 其他依赖包
pip install transformers
pip install einops

# install git
apt-get install git
git config --global user.name "your_name"
git config --global user.email "***@qq.com"

# 加速编译的包
pip install ninja
pip install packaging

# 融合算子包
# flash-attn, 2.1.0支持torch>=1.12, cuda>=11.4
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout tags/v2.1.0
python setup.py install

# rotary_emb
cd /root/software/flash-attention/csrc/rotary
pip install .

# xentropy_cuda_lib
cd /root/software/flash-attention/csrc/xentropy
pip install .

# dropout_layer_norm, cuda11.4下安装会卡住，可以忽略这个包，然后用apex
# cd /root/software/flash-attention/csrc/layer_norm
# pip install .

# apex
cd /root/software
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout -b 22.04-dev  origin/22.04-dev
# ----- vim setup.py, and remove the code "check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)" of setup.py
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./

# xformers, 0.0.23支持torch>=1.12, cuda>=11.4
cd /root/software
git clone https://github.com/facebookresearch/xformers.git
cd xformers
git checkout tags/v0.0.23
# 导入子模块中的第三方仓库
git submodule update --init --recursive
python setup.py install  # 安装完后的是xformers0.0.24, 不过不影响

# 最后删除这些文件夹
vim ~/.bashrc
# ---------- set auto workspace
cd ~
# ---------- wq

#$ in your system
# 提交为镜像
docker commit container_id llm_2.0.1_11.4
# 1. 上传到dockerhub中
docker login
docker tag llm_2.0.1_11.4 stdiohg/llm:pytorch2.0.1-cuda11.4
docker push stdiohg/llm:pytorch2.0.1-cuda11.4
# 2. 或者打包为压缩包
docker save -o llm_pytorch201_cuda114.tar image_id
# 3. 或上传到阿里镜像仓库
# 见：https://cr.console.aliyun.com/repository/cn-hangzhou/stdiohg/llm/details
```
