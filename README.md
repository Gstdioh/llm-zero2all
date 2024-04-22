# llm-zero2all

从零开始编写与大语言模型有关的所有代码，用于学习

## 0 数据集构建
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

`json文件
{
    gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str,
    others: dict
}`

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

3. tokenizers库中的BPE没有实现byte_fallback，需要使用ByteLevel才行

4. 通常，char-BPE可以使用NFKC，Byte-BPE一般不用

5. tokenizers库中BPE下使用ByteLevel不会先构建256的词表，ByteLevelBPETokenizer会，所以要用BBPE的话，用ByteLevelBPETokenizer，其中可以改为自己的正则表达式来预分割

6. 训练tokenizer时，训练集大小：20G，分成单位为200MB的txt文件
    1. hf， 占用内存：288G
    2. spm，占用内存：60G

### 分词器（Tokenizer）

## 1 构建基础语言模型 Z2all (使用 llama2 结构)
