import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import json
import argparse

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import get_file_paths


# TRAIN_DATA_DIR = "./data/02_train_data"  # 在这个文件夹中找到所有的json文件
TRAIN_DATA_DIR = "./data/01_pre_data"  # 在这个文件夹中找到所有的json文件
SAVE_DIR = os.path.join(TRAIN_DATA_DIR, "01_bin_for_train_")  # bin保存的目录

MAX_WORKERS = 16
START_TEXT = "json_"

def process_file(args, tokenizer_dir):
    """json文件，一行一个json对象
    { gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str, others: dict }
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    
    file_id, file_path = args
    
    # 判断是json还是jsonl文件
    is_jsonl = False
    with open(file_path, "r") as f:
        if f.read(1) == "[":
            is_jsonl = True
    
    all_token_ids = []
    with open(file_path, "r") as f:
        if is_jsonl:
            all_lines = json.load(f)
        else:
            all_lines = f.readlines()
        for line in tqdm(all_lines, desc=f"处理文件 {file_path}", position=file_id):
            if is_jsonl:
                data = line
            else:
                data = json.loads(line)
                
            # 文本内容
            text = ""
            try:
                if len(data["title"] + data["desc"]) == 0:
                    text = data["content"]
                else:
                    text = data["title"] + data["desc"] + "\n" + data["content"]
            except:
                text = data["content"]
            text = text.strip()  # 去除首尾空格
            # 编码文本，使用<|beginoftext|>
            token_ids = tokenizer.encode(text, add_begin=True)
            all_token_ids.extend(token_ids)
            
    # 转换为unint16类型的numpy数组，因为token的范围是[0, 65535]，从而减小bin文件的内存占用
    all_token_ids = np.array(all_token_ids, dtype=np.uint16)
    
    # 保存tokenized文件到bin目录
    file_basename = os.path.basename(file_path)
    bin_basename = file_basename.replace(".json", "").replace("json_", "") + f"_{file_id:04}.bin"
    tokenized_filename = os.path.join(SAVE_DIR, bin_basename)

    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_token_ids.tobytes())
    # calculate the average sequence length (they are separated by <|beginoftext|>)
    begin_id = tokenizer.special_tokens["<|beginoftext|>"]
    avg_seq_len = all_token_ids.size / ((all_token_ids == begin_id).sum())
    print(f"Saved {tokenized_filename}, token_id average seqlen: {avg_seq_len:.2f}")


def pretokenize_data(tokenizer_dir):
    # tokenize所有的训练文件，碎片化的json文件
    file_paths = get_file_paths(TRAIN_DATA_DIR, "json", start_text=START_TEXT)
    
    # 在单个进程中测试下
    # process_file((0, file_paths[0]), tokenizer_dir)
    
    # 在一个进程池中处理所有文件
    fun = partial(process_file, tokenizer_dir=tokenizer_dir)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(fun, enumerate(file_paths))
    print("Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer_type", type=str, default="hf", help="hf or sp")
    args = parser.parse_args()
    
    tokenizer_dir = f"./tokenizer/{args.tokenizer_type}_bbpe_tokenizer"
    
    # tokenize后的bin目录
    SAVE_DIR = SAVE_DIR + args.tokenizer_type
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    pretokenize_data(tokenizer_dir)
