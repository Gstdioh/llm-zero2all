# llm-zero2all

从零开始编写与大语言模型有关的所有代码，用于学习

## 00 数据集构建
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

## 01 分词器（Tokenizer）

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

## 01 构建基础语言模型 Z2all (使用 llama2 结构)
