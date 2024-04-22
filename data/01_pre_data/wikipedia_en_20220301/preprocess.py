import os
import glob

import json
from tqdm import tqdm
import pandas as pd

data_dir = "./"
save_dir = "./txt_wikipedia_en_20220301"
file_prefix = "txt_wikipedia_en_20220301_"
file_text_len = 10e6  # 英文，大概10MB, *20=200MB
file_text_len = 200e6
os.makedirs(save_dir, exist_ok=True)

# 获取当前文件夹下的所有文，除了valid数据集
filenames = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
filenames = [filename for filename in filenames if not filename.endswith("valid.json")]

# 循环读取所有文件
texts = []
size_sum = 0
count = 0
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as f:
        # 读取所有行
        df = pd.read_parquet(filename)
        for _, data in tqdm(df.iterrows(), desc=f'处理"{filename}"', total=df.shape[0]):
            # 防止字段不存在
            try:
                texts.append(data["title"] + ". " + data["text"] + "\n")
            except KeyError:
                continue
            size_sum += len(texts[-1])
            
            if size_sum > file_text_len:
                # 将数据写入到新文件中
                with open(os.path.join(save_dir, file_prefix + f"{int(count):04}.txt"), "w", encoding="utf-8") as ftxt:
                    ftxt.write(''.join(texts))
                texts = []
                size_sum = 0
                count += 1
