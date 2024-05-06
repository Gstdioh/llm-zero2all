# llm-zero2all

从零开始编写与大语言模型有关的所有代码，用于学习

## 00 环境配置
我的环境：cuda11.4, pytorch1.12.1

见 requirements.txt，融合算子相应库的具体安装可以看[03-融合算子小节](#融合算子)

### 硬件
查看卡间通信：`nvidia-smi topo -m`

GPU通信方式：https://zhuanlan.zhihu.com/p/74217534

## 01 数据集构建
### 包含的数据集
| 类别  | 文件名                     | raw大小 | 存储格式                 | json大小 | txt大小 | train |
| ----- | ------------------------- | ------- | ----------------------- | -------- | ------- | ----- |
| 中文  | baike2018qa               | 1.5G    | .json, 一行一json对象    | 1.5G, 8  | 1.3G    | 8/8   |
|       | new2016zh                 | 8.6G    | .json, 一行一json对象    | 8.5G, 44 | 7.5G    | 20/44 |
|       | webtext2019zh             | 3.8G    | .json, 一行一json对象    | 4.0G, 21 | 3.1G    | 21/21 |
|       | wikipedia_cn_20240418     | 2.5G    | .json, 一行一json对象    | 2.6G, 14 | 2.5G    | 14/14 |
|       | Zhihu-KOL                 | 1.4G    | .parquet                |          |         |       |
|       | WuDaoCorpus2.0_base_200G  | 201G    | .json, 一文件一json对象  |          |         |       |
| 英文  | wikipedia_en_20220301     | 11G     | .parquet                 | 20G, 103 | 19G     | 40/101|
| 代码  | github-python             | 2.1G    | .json, 一行一json对象    | 2.2G, 11 | 2.0G    | 11/11 |

**json文件**，一行一个json对象

`
{ gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str, others: dict }
`

### 命令
1. `hfd.sh`，下载huggingface模型和数据集, (--model, dataset)

    `./hfd.sh wikitext --dataset --tool aria2c -x 4 --include data/20220301.en`

2. `unzip`, 解压.zip

    `unzip file.zip -d ./data`

3. `unrar`, 解压.rar, x表示解压, 自动创建文件夹

    `unrar x file.rar`

4. `tar`, 解压.tar.gz

    `tar -xzf file.tar.gz`

5. `gunzip`, 解压.json.gz为.json

    `gunzip file.json.gz`

6. `du`, 显示文件大小，s不递归显示，h展示友好，以block-size为单位

    `du -sh * --block-size=1M`

### 注意
1. `data_str += json.dumps(data, ensure_ascii=False) + "\n"  # 注意加上ensure_ascii=False，否则会保存为Unicode编码`

2. chr(2581)表示unicode码位为2581整数，'\u2581'表示4位16进制的unicode编码，注意进制不同

3. 训练tokenizer时，训练集大小：20G，分成单位为200MB的txt文件
    1. hf， 占用内存：288G
    2. spm，占用内存：60G，错
    训练集大小：3G， hf ，txt，内存峰值：97G
    训练集大小：10G，hf ，txt，内存峰值：160G，训练时间22小时
    训练集大小：5G， spm，txt，爆
    训练集大小：4G， spm，txt，可
    训练集大小：20G，spm，txt，内存峰值：135G，训练时间22小时

4. 可以通过 `sys.path.append("../")` 来添加包的搜索路径

5. 通过`f.seek(0,2)`移动读写位置（offset: 0表示偏移，whence: 2表示文件末尾）和`f.tell()`获取当前读写位置，可以快速获得当前文件的bytes大小

6. 使用torchrun时，因为我的解释器路径太长而被截断了。。。
可以创建一个符号链接指向原python解释器，缩短路径长度，注意将路径内容改为自己的：

    `ln -s /path/to/raw/python /path/to/link`

    然后将torchrun脚本第一行从`#!/path/to/raw/python`改为`#!/path/to/link`

### 流式读取文件

#### json
```python
def stream_json(file_path):
    with open(file_path, 'r') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            yield obj
```

#### jsonl
```python
def stream_jsonl(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            yield obj
```

#### parquet
见：https://stackoverflow.com/questions/68819790/read-write-parquet-files-without-reading-into-memory-using-python
```python
if file_path.endswith(".parquet"):
    # df = pd.read_parquet(file_path)  # 1.8GB
    # 流式读取，节省内存
    batch = ParquetFile(file_path)  # 0.5MB
    record = batch.iter_batches(
        batch_size=10,
        columns=["content"],
    )
    # df = pd.read_parquet(file_path)
    for lines in record:
        lines = lines.to_pydict()
        for text in lines["content"]:
            text = text.strip()  # 去除首尾空格
            yield text
```

## 02 分词器（Tokenizer）

### 1 训练自己的tokenizer

使用huggingface的tokenizers库或者sentencepiece库来构建分词器

#### 1.1 tokenizers库

具体使用方法见[tokenizers官方文档](https://huggingface.co/docs/tokenizers)

代码见train_tokenizer.py，注意事项：

* tokenizers库中的BPE没有实现byte_fallback，需要使用ByteLevel才行

* 通常，char-BPE可以使用NFKC，Byte-BPE一般不用

* tokenizers库中BPE下使用ByteLevel不会先构建256的词表，ByteLevelBPETokenizer会，所以要用BBPE的话，用ByteLevelBPETokenizer，其中可以改为自己的正则表达式来预分割

* tokenizers库中使用encode会区分特殊token（即不会对特殊token进行分割），这会导致用户能在文本中写入侵入文本（即能控制特殊token，会有一定的风险），但是我不知道怎么关闭这个功能，所以我把训练后的json中的added_token全部删除了，然后在构建自己的tokenizer类时自己来管理特殊token（缺点是效率可能会低点）

#### 1.2 sentencepiece库

具体使用方法见：[spm官方的python使用方法](https://github.com/google/sentencepiece/blob/master/python/README.md)

详细参数见：[scpm官方，训练所需参数](https://github.com/google/sentencepiece/blob/master/doc/options.md)

其他资料：[LLM大模型之基于SentencePiece扩充LLaMa中文词表实践](https://zhuanlan.zhihu.com/p/655281268)

### 2 使自己的tokenizer可以通过save_pretrained(dir)保存，通过AutoTokenizer.from_pretrained(dir)导入

参考chatglm-6b和Qwen-7B的构建代码和官方 (AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast) 的源码

chatglm使用sentencepiece，https://huggingface.co/THUDM/chatglm-6b/blob/main/tokenization_chatglm.py

Qwen使用tiktoken，其自己管理特殊token，https://huggingface.co/Qwen/Qwen-7B/blob/main/tokenization_qwen.py

其他资料：[huggingface AutoTokenizer.from_pretrained流程](https://zhuanlan.zhihu.com/p/621106604)

下面介绍我总结的方法：

1. 构建一个类，继承PreTrainedTokenizer，然后通过上述训练得到的tokenizer，实现自己的encode和decode
2. 初始化，需要实现父类PreTrainedTokenizerBase的一个方法，在 `super.__init__(**kwargs)` 时会调用该方法
    ```python
    def get_vocab(self) -> Dict[str, int]:
        """
        返回词汇表
        """
        return self.tokenizer.get_vocab()
    ```
3. 要使用save_pretrained(dir)，需要实现父类两个方法
    ```python
    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.tokenizer.vocab_size
    
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> Tuple[str]:
        """
        保存tokenizer所需要的文件
        """
        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
        )
        self.tokenizer.backend_tokenizer.save(tokenizer_file)
        
        return (tokenizer_file,)
    ```
    save_pretrained(dir)后会在dir构建三个文件：special_tokens_map.json, tokenizer_config.json, tokenizer.json
    
    注意，为了在from_pretrained(dir)时能正确加载，需要手动添加auto_map的信息，如下
4. 要使用AutoTokenizer.from_pretrained(dir)，需要1或2
    1. 要么在初始化时，添加如下代码，添加映射保证AutoTokenizer.from_pretrained(dir)可以加载（推荐）
        ```python
        self._auto_map = { "AutoTokenizer": ["my_hf_tokenizer.MyHFTokenizer", None] }
        ```
        其会在save_pretrained(dir)时，在tokenizer_config.json中自动添加auto_map
    2. 要么手动添加一个tokenizer_config.json文件（名字固定），内容为：
        ```json
        {
            "tokenizer_class": "MyHFTokenizer",
            "auto_map": {
                "AutoTokenizer": [
                "my_hf_tokenizer.MyHFTokenizer",
                null
                ]
            }
        }
        ```
    3. 注意，AutoTokenizer.from_pretrained(dir)实际调用的是PreTrainedTokenizerBase.from_pretrained(dir)

        通过搜索dir中的vocab_files文件，通过init_kwargs来传递到自己tokenizer类的__init__()中

        `vocab_files = {**cls.vocab_files_names, **additional_files_names}`

        所以属性名要对应好，例如，属性名是vocab_file，则要构建相应的字典作为类属性
        ```python
        class QWenTokenizer(PreTrainedTokenizer):
            """QWen tokenizer."""

            vocab_files_names = {"vocab_file": "qwen.tiktoken"}  # 此处key和下面的属性名对应

            def __init__(
                self,
                vocab_file,  # 对应上面，实际vocab_file = "dir/qwen.tiktoken"，即作用是获得完整路径
                errors="replace",
                extra_vocab_file=None,
                **kwargs,
            ):
        ```
        注意additional_files_names自动包含了一些要搜索的文件
        ```python
        additional_files_names = {
            "added_tokens_file": ADDED_TOKENS_FILE,  # kept only for legacy
            "special_tokens_map_file": SPECIAL_TOKENS_MAP_FILE,  # kept only for legacy
            "tokenizer_config_file": TOKENIZER_CONFIG_FILE,
            # tokenizer_file used to initialize a slow from a fast. Properly copy the `addedTokens` instead of adding in random orders
            "tokenizer_file": FULL_TOKENIZER_FILE,
        }
        ```

## 03 构建基础语言模型 Z2all (使用 llama2 结构)

### 融合算子
包括flash-attn, rope, cross_entropy, rmsnorm, swiglu
训练设置：1个A800-80G，每个迭代训练的token数：

tokens per iteration will be: 262,144

breaks down as: 16 grad accum steps * 1 processes * 2 batch size * 2048 max seq len

训练命令：`python pretrain.py --batch_size=2 --gradient_accumulation_steps=16`

模型和优化器状态占用内存：12250MB = 12.25GB

每个iter，性能比较：

| 融合算子           | 速度比较/s | MFU (Model FLOPs Utilization)/% | 内存占用/GB |
| :---------------: | :-------: | :----------------------------: | :---------: |
| navie             | 9.85      | 18.41                          | 71.32       |
| use_all           | 3.23      | 56.06                          | 27.87       |
| w/o flash-attn    | 8.69      | 20.87                          | 65.43       |
| w/o rope          | 3.51      | 51.70                          | 27.90       |
| w/o cross_entropy | 3.37      | 53.70                          | 29.83       |
| w/o rmsnorm       | 3.78      | 47.94                          | 30.68       |
| w/o swiglu        | 3.60      | 50.39                          | 29.79       |

---

通常会需要ninja和packagin这两个包来编译，ninja用来加速编译
```bash
# 加速安装
pip install ninja
pip install paskaging
```

#### 1 flash-attn
```bash
# 2.1.0需要pytorch>=1.12, cuda>=11.4
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout tags/v2.1.0

# 不要用`pip install .`安装
python setup.py install
```

#### 2 rotary_emb
```bash
# 继续在flash-attention中
cd flash-attention/cscr/rotary
pip install .
```

#### 3 xentropy_cuda_lib
```bash
# 继续在flash-attention中
cd flash-attention/cscr/xentropy
pip install .
```

#### 4 dropout_layer_norm
```bash
# 继续在flash-attention中，注意在cuda11.4下安装会卡住，同时pytorch2.1.0下安装有问题，版本不兼容
# 我直接用的qwenllm/qwen latest镜像，已经安装好了，环境是pytorch2.0.1和cuda11.7
# 但是我的训练环境因为只有cuda11.4驱动，也不好更新驱动，所以实际训练时没有用这个包
cd flash-attention/cscr/layer_norm
pip install .
```

#### 5 MixedFusedRMSNorm (apex)
apex中有MixedFusedRMSNorm，如果安装不了上面的dropout_layer_norm，可以用这个来加速

在 cuda11.4, pytorch1.12.1_cuda11.3 环境下安装时，需要注释掉apex的setup.py中的 `check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)`

安装见：https://zhuanlan.zhihu.com/p/672284687
```bash
# 获取apex, 注意这里一定要通过git clone 不要自己下载zip包，不然就会碰到错误4
git clone https://github.com/NVIDIA/apex.git

cd apex
git branch -a
git checkout -b 22.04-dev  origin/22.04-dev #切换分支，当前的主分支有问题，你会碰到错误5

pip uninstall apex  #卸载之前的apex

# 编译apex，要加 --disable-pip-version-check 不然你会碰到错误2
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" ./

等待10分钟....
```

#### 6 SwiGLU (xformers)
这个算子在xformers库中实现，注意需要ninja来加速编译

```bash
# v0.0.23时，torch >= 1.12
git clone https://github.com/facebookresearch/xformers.git
git checkout tags/v0.0.23

# 导入子模块中的第三方仓库
git submodule update --init --recursive

python setup.py install
```

注意，在xformers v0.0.23中，如果要使用autocast，那么要删除xformers/ops/swiglu_op.py中`_ForwardToPythonAutogradFunc`中的这一段代码：

```python
if op.dtype_autocast_gpu == torch.bfloat16:
    return False
```

因为在pytorch1.12.1版本以下有一个bug，见`./test_pytorch_bug.py`文件，主要是使用autocast时，反向传播会将bfloat16错误转换为float16，你可以升级pytorch，或者根据[链接](https://github.com/pytorch/pytorch/commit/bc03aa6013e101222c9652d04a2b08e48f626dfb#diff-dac4bd53ced015c8810b3b02fc5c2ec6c2b0658c5090b4fbbd09c96bd45087d1)
来修改你的pytorch源码。

## 04 训练

### 使用pytorch进行DP训练

#### 训练超参数设置

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

#### 注意

1. 使用nccl通信后端，代码在DDP()包裹模型时会卡住，通过禁用P2P通信解决，见：https://github.com/pytorch/pytorch/issues/23074，但是会影响通信效率。或者可以改为gloo通信后端。

    GPU通信方式：https://zhuanlan.zhihu.com/p/74217534