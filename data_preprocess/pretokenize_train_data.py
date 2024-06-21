"""
1. 添加了流式读取方式，减少内存占用
2. 可以同时处理json, jsonl, parquet文件
3. 支持多进程处理
4. 将所有文件平均分割为100MB大小的bin文件
"""

import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import json
import argparse
import random

import numpy as np
import ijson
import pandas as pd
from pyarrow.parquet import ParquetFile
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import get_file_paths, is_json_file


def stream_json(file_path):
    with open(file_path, 'r') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            yield obj


def stream_jsonl(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            yield obj


def file_text_generator(file_path):
    """
    流式生成文件中每一行数据的text
    """
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
    elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
        # 开始流式读取
        if is_json_file(file_path):
            lines = stream_json(file_path)
        else:
            lines = stream_jsonl(file_path)
        for data in lines:
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
            yield text


def process_file(args, tokenizer_dir):
    """
    每个进程处理多个文件
    
    json文件，一行一个json对象
    { gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str, others: dict }
    parquet文件, key, 只关注content列
    ['dataType', 'title', 'content', 'uniqueKey', 'titleUkey', 'id']
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    
    process_id, file_paths = args
    file_paths = file_paths.tolist()
    
    file_id = 0
    all_token_ids = []
    for file_path in tqdm(file_paths, desc=f"处理文件 {file_paths[0]}", position=process_id):
        for text in file_text_generator(file_path):
            # 编码文本，使用<|beginoftext|>
            token_ids = tokenizer.encode(text, add_begin=True)
            all_token_ids.extend(token_ids)
            
            if len(all_token_ids) >= FILE_BYTES / ELEMENT_SIZE:
                # 转换为unint16类型的numpy数组，因为token的范围是[0, 65535]，从而减小bin文件的内存占用
                all_token_ids = np.array(all_token_ids, dtype=np.uint16)
                
                # 保存tokenized文件到bin目录
                bin_basename = f"file_{process_id:03}_{file_id:05}.bin"
                tokenized_filename = os.path.join(SAVE_DIR, bin_basename)
                
                # write the bytes
                with open(tokenized_filename, "wb") as f:
                    f.write(all_token_ids.tobytes())
                # calculate the average sequence length (they are separated by <|beginoftext|>)
                begin_id = tokenizer.special_tokens["<|beginoftext|>"]
                avg_seq_len = all_token_ids.size / ((all_token_ids == begin_id).sum())
                print(f"Saved {tokenized_filename}, token_id average seqlen: {avg_seq_len:.2f}")
                
                all_token_ids = []
                file_id += 1
    
    # 记得最后剩余的数据
    if len(all_token_ids) > 0:
        # 转换为unint16类型的numpy数组，因为token的范围是[0, 65535]，从而减小bin文件的内存占用
        all_token_ids = np.array(all_token_ids, dtype=np.uint16)
        
        # 保存tokenized文件到bin目录
        bin_basename = f"file_{process_id:03}_{file_id:05}.bin"
        tokenized_filename = os.path.join(SAVE_DIR, bin_basename)
        
        # write the bytes
        with open(tokenized_filename, "wb") as f:
            f.write(all_token_ids.tobytes())
        # calculate the average sequence length (they are separated by <|beginoftext|>)
        begin_id = tokenizer.special_tokens["<|beginoftext|>"]
        avg_seq_len = all_token_ids.size / ((all_token_ids == begin_id).sum())
        print(f"Saved {tokenized_filename}, token_id average seqlen: {avg_seq_len:.2f}")
        
        all_token_ids = []
        file_id += 1


def pretokenize_data(tokenizer_dir):
    # tokenize所有的训练文件，碎片化的json文件
    file_paths = get_file_paths(TRAIN_DATA_DIR, FILE_TYPE, start_text=START_TEXT)
    
    rng = random.Random(42)
    rng.shuffle(file_paths)
    
    # 将file_paths列表平均分为MAX_WORKERS份
    file_paths_list = np.array_split(file_paths, MAX_WORKERS)
    
    # 在单个进程中测试下
    # process_file((0, file_paths_list[0]), tokenizer_dir)
    
    # 在一个进程池中处理所有文件
    fun = partial(process_file, tokenizer_dir=tokenizer_dir)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(fun, enumerate(file_paths_list))
    print("Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/02_train_data_more", help="要pretokenize的文件目录，将其中的所有文件tokenize后保存到bin目录中，可以识别parquet、json和jsonl文件，注意字段可能不同，需要自己设置")
    parser.add_argument("--tokenizer_dir", "--tokenizer_type", type=str, default="tokenizer/hf_bbpe_tokenizer", help="tokenizer的路径")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    tokenizer_dir = args.tokenizer_dir
    
    TRAIN_DATA_DIR = data_dir  # 在这个文件夹中找到所有的数据文件
    FILE_TYPE = ["parquet", "json", "jsonl"]  # 文件类型
    SAVE_DIR = os.path.join(TRAIN_DATA_DIR, "01_bin_for_train")  # bin保存的目录

    FILE_BYTES = 100 * 1024 * 1024  # 每个文件的字节数, 100MB
    ELEMENT_SIZE = 2  # 每个元素的字节数
    MAX_WORKERS = 16
    START_TEXT = ""
    
    # tokenize后的bin目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    pretokenize_data(tokenizer_dir)
