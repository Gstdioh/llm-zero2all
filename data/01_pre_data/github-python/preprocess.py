import os
import glob

import json
from tqdm import tqdm

data_dir = "./"
save_dir = "./txt_github-python"
file_prefix = "txt_github-python_"
file_text_len = 80e6  # 对于coda，大概76MB
file_text_len = 220e6
os.makedirs(save_dir, exist_ok=True)

# 获取当前文件夹下的所有文，除了valid数据集
filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
filenames = [filename for filename in filenames if not filename.endswith("valid.json")]

# 循环读取所有文件
texts = []
size_sum = 0
count = 0
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as f:
        # 读取所有行
        lines = list(tqdm(f, desc=f'构建"{filename}"文件列表'))
        lines = tqdm(lines, desc=f'处理"{filename}"')
        for line in lines:
            # 将每一行转换为json格式
            data = json.loads(line)
            # 防止字段不存在
            try:
                texts.append(data["content"] + "\n")
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
