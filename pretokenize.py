import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from tqdm import tqdm

from utils import get_file_paths
from tokenizer import Tokenizer

TRAIN_DATA_DIR = "./data/02_train_data"
PREFIX = "01_bin_for_train_tok"

def process_file(args, vocab_size):
    file_id, file_path = args
    
    # 构建tokenizer
    tokenizer_model = "my_hf_bbpe_tokenizer.json"
    tokenizer = Tokenizer(tokenizer_model)
    
    # json文件，一行一个json对象
    """json文件
    { gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str, others: dict }
    """
    all_tokens = []
    with open(file_path, "r") as f:
        data = json.loads(f.readline())
        # 文本内容
        text = ""
        if len(data["title"] + data["desc"]) == 0:
            text = data["content"]
        else:
            text = data["title"] + data["desc"] + "\n" + data["content"]
        text = text.strip()  # 去除首尾空格
        # 编码文本，使用BOS
        tokens = tokenizer.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    
    # 转换为unint16类型的numpy数组，因为token的范围是[0, 65535]，从而减小bin文件的内存占用
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    
    # 保存tokenized文件到bin目录
    bin_dir = os.path.join(TRAIN_DATA_DIR, f"{PREFIX}{vocab_size}")
    file_basename = os.path.basename(file_path)
    bin_basename = file_basename.replace(".json", ".bin")
    tokenized_filename = os.path.join(bin_dir, bin_basename)

    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize(vocab_size):
    # tokenize所有的训练文件，碎片化的json文件
    file_paths = get_file_paths(TRAIN_DATA_DIR, "json")
    
    # tokenize后的bin目录
    bin_dir = os.path.join(TRAIN_DATA_DIR, f"{PREFIX}{vocab_size}")
    os.makedirs(bin_dir, exist_ok=True)

    # 在一个进程池中处理所有文件
    fun = partial(process_file, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(file_paths))
    print("Done.")
    

if __name__ == "__main__":
    pretokenize(vocab_size=64000)
