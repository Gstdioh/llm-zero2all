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

from utils import get_file_paths, get_file_line_count
from sft import Conversation


TRAIN_DATA_DIR = "./data/04_sft_conversation_data"  # 在这个文件夹中找到所有的数据文件
FILE_TYPE = ["json"]  # 文件类型
SAVE_DIR = os.path.join(TRAIN_DATA_DIR, "01_bin_for_sft_")  # bin保存的目录

input_ids_suffix = "_input_ids.bin"
labels_suffix = "_labels.bin"
sample_index_map_suffix = "_sample_index_map.ijson"  # ijson表示用于index的json文件

USER_NAME = "human"
ASSISTANT_NAME = "assistant"
IGNORE_INDEX = -100

SAVE_DTYPE = np.uint16

MAX_WORKERS = 8
START_TEXT = ""


def stream_json(file_path):
    with open(file_path, 'r') as f:
        objects = ijson.items(f, 'item')
        for obj in objects:
            yield obj


def stream_jsonl(file_path):
    # test_count = 5000
    with open(file_path, 'r') as f:
        for line in f:
            obj = json.loads(line)
            yield obj
            # test_count -= 1
            # if test_count == 0:
            #     break


def file_data_generator(file_path):
    """
    流式生成文件中每一行的数据
    """
    # 判断是json还是jsonl文件
    is_json = False
    with open(file_path, "r") as f:
        if f.read(1) == "[":
            is_json = True
    
    # 开始流式读取
    if is_json:
        return stream_json(file_path)
    else:
        return stream_jsonl(file_path)


def process_file(args, tokenizer_dir):
    """
    每个进程处理多个文件
    
    多轮对话指令微调数据集的预处理，进行prompt构建、tokenize等操作
    
    file_paths样例，jsonl格式：
    {
        "id": "identity_0",
        "conversations": [
            {
                "from": "human",
                "value": "Who are you?"
            },
            {
                "from": "assistant",
                "value": "I am a helpful assistant."
            },
            {
                "from": "human",
                "value": "Have a nice day!"
            },
            {
                "from": "assistant",
                "value": "You too!"
            }
        ]
    }
    ...
    
    保存三个文件：
    1. input_ids.bin: 保存所有的input_ids
    2. labels.bin: 保存所有的labels
    3. sample_index_map.json: 保存每个样本的起始偏移和长度
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    
    conv = Conversation(tokenizer=tokenizer)

    name_convert = {
        USER_NAME: conv.roles_name[0],
        ASSISTANT_NAME: conv.roles_name[1]
    }
    
    process_id, file_paths = args
    file_paths = file_paths.tolist()
    
    for file_path in file_paths:
        # 新的基本文件名：data/04_sft_conversation_data/01_bin_for_sft_hf_2048/train_3.5M_CN
        new_base_file_path = os.path.join(SAVE_DIR, ".".join(os.path.basename(file_path).split(".")[:-1]))
        
        # 将file_path的后缀改为.bin
        # splitext() 方法用于分离文件名与扩展名
        # 保存的sample数据
        save_file_path_input_ids = new_base_file_path + input_ids_suffix
        save_file_path_labels = new_base_file_path + labels_suffix
        # 保存的sample_index_map的数据
        save_file_path_sample_index_map = new_base_file_path + sample_index_map_suffix
        sample_index_map = []
        sample_start_offset = 0  # 某个sample的起始偏移
        sample_len = 0
        
        # 快速获取文件的行数
        file_line_count = get_file_line_count(file_path)
        
        # 存在则删除
        if os.path.exists(save_file_path_input_ids):
            os.remove(save_file_path_input_ids)
        if os.path.exists(save_file_path_labels):
            os.remove(save_file_path_labels)
        if os.path.exists(save_file_path_sample_index_map):
            os.remove(save_file_path_sample_index_map)
        
        # 遍历某个文件中的所有行
        for data in tqdm(file_data_generator(file_path), desc=f"[Process {file_path}]", total=file_line_count, position=process_id):
            conversations = data["conversations"]
            
            if name_convert[conversations[0]["from"]] != conv.roles_name[0]:
                # Skip the first one if it is not from human
                conversations = conversations[1:]

            conv.clear()  # 清空messages
            for j, sentence in enumerate(conversations):
                role = name_convert[sentence["from"]]
                assert role == conv.roles_name[j % 2], f"Get a wrong role {role} at turn {j}, expected {conv.roles_name[j % 2]} at turn {j}"
                conv.append_message(role, sentence["value"])
            
            # 构建prompt，并且进行tokenize，会对labels进行掩码
            output = conv.get_tokenized_prompt(return_type="np", ignore_index=IGNORE_INDEX, add_begin=True)
            
            # 这里进行max_seq_len的截断，后面读取数据集的时候再根据需求截断
            input_ids = output["input_ids"]
            labels = output["labels"]
            
            # 添加一个样本的索引
            sample_len = len(input_ids)
            sample_index_map.append((sample_start_offset, sample_len))
            sample_start_offset += sample_len
            
            with open(save_file_path_input_ids, "ab") as f:
                f.write(input_ids.astype(SAVE_DTYPE).tobytes())
            with open(save_file_path_labels, "ab") as f:
                f.write(labels.astype(SAVE_DTYPE).tobytes())

        # 保存sample_index_map
        with open(save_file_path_sample_index_map, "w", encoding="utf-8") as f:
            json.dump(sample_index_map, f)


def gather_all_samlpe_index_map(file_paths):
    """
    聚合所有的索引，原始的是：[(sample_start_offset, sample_len), ...]
    聚合后：[(file_id, sample_start_offset, sample_len)], ...]
    """
    all_sample_index_map = []
    
    print("Gathering all sample index map...")
    for file_path in file_paths:
        new_base_file_path = os.path.join(SAVE_DIR, ".".join(os.path.basename(file_path).split(".")[:-1]))
        sample_index_map_path = new_base_file_path + sample_index_map_suffix
        with open(sample_index_map_path, "r", encoding="utf-8") as f:
            sample_index_map = json.load(f)
        
        file_id = len(all_sample_index_map)
        all_sample_index_map.extend([(file_id, sample_start_offset, sample_len) for sample_start_offset, sample_len in sample_index_map])
    
    # ijson表示用于index的json文件
    with open(os.path.join(SAVE_DIR, "all_sample_index_map.ijson"), "w", encoding="utf-8") as f:
        json.dump(all_sample_index_map, f)


def pretokenize_sft_data(tokenizer_dir):
    # tokenize所有的训练文件，碎片化的json文件
    file_paths = get_file_paths(TRAIN_DATA_DIR, FILE_TYPE, startswith=START_TEXT)
    
    max_workers = min(MAX_WORKERS, len(file_paths))
    
    # 将file_paths列表平均分为MAX_WORKERS份
    file_paths_list = np.array_split(file_paths, max_workers)
    
    # 在单个进程中测试下
    # process_file((0, file_paths_list[0]), tokenizer_dir)
    
    # 在一个进程池中处理所有文件
    fun = partial(process_file, tokenizer_dir=tokenizer_dir)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(fun, enumerate(file_paths_list))

    gather_all_samlpe_index_map(file_paths)
        
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tokenizer_type", type=str, default="hf", help="hf or sp")
    args = parser.parse_args()
    
    tokenizer_dir = f"./tokenizer/{args.tokenizer_type}_bbpe_tokenizer"
    
    # tokenize后的bin目录
    SAVE_DIR = SAVE_DIR + args.tokenizer_type
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    pretokenize_sft_data(tokenizer_dir)
