import os
import sys
import random

import numpy as np
from tqdm import tqdm

sys.path.append("../")
from utils import get_file_paths


seed = 42
rng = random.Random(seed)

files_dir = "./02_train_data/01_bin_for_train_hf"
file_paths = get_file_paths(files_dir, file_type="bin")

fetch_size = 1 * 1024 * 1024  # 每个文件取1M，取小点，让valid更具代表性，实际是fetch_size//2个uint16
fetch_count = (128 * 1024 * 1024) // fetch_size  # 一共取的文件数
single_valid_count = (128 * 1024 * 1024) // fetch_size  # 几个文件合在一块

valid_data_dir = "./02_train_data/02_bin_for_valid_hf"
valid_prefix = "valid_"
os.makedirs(valid_data_dir, exist_ok=True)

i = 0
count = 0
all_valid_data = []
# 取fetch_count个文件，每个文件取fetch_size//2个uint16
for _ in tqdm(range(fetch_count)):
    rng.shuffle(file_paths)
    file_path = file_paths[0]

    data = np.fromfile(file_path, dtype=np.uint16)
    
    valid_data = data[:fetch_size//2]
    residual_data = data[fetch_size//2:]
    
    # 将剩余的数据写回原文件
    with open(file_path, "wb") as f:
        f.write(residual_data.tobytes())
    
    all_valid_data.append(valid_data)
    count += 1
        
    if count % single_valid_count == 0:
        all_valid_data = np.concatenate(all_valid_data)

        with open(os.path.join(valid_data_dir, f"{valid_prefix}{i:04}.bin"), "wb") as f:
            f.write(all_valid_data.tobytes())
        i += 1
        all_valid_data = []
