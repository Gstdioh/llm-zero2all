import os
import glob
import regex as re
import unicodedata

import json
from tqdm import tqdm

# 不需要清洗了，因为在转json时就清洗过了
def clean_text(s):
    # 保留"\n"和所有可显示的字符
    s = ''.join(c for c in s if unicodedata.category(c) not in ["Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"] or c == '\n')
    s = s.strip()  # 去除首尾空格
    # s = unicodedata.normalize('NFKC', s)  # 规范化，将全角字符转换为半角字符，弃用：在tokenizer中进行规范化；或者不进行规范化
    return s

data_src = "github-python"
file_type = ".txt"

# 将该文件夹的数据转换为txt文件
data_dir = f"./json_{data_src}"

save_dir = f"./txt_{data_src}"
file_prefix = f"txt_{data_src}_"
file_bytes = 200 * 1024 * 1024
os.makedirs(save_dir, exist_ok=True)

# 获取当前文件夹下的所有文件，除了valid数据集
filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

# 循环读取所有文件
buffer_data = []  # 一个文件的数据
sum_bytes = 0  # 
count = 0
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as fsrc:
        # 读取所有行
        lines = list(tqdm(fsrc, desc=f'构建"{filename}"文件列表'))
        lines = tqdm(lines, desc=f'处理"{filename}"')
        for line in lines:
            # 将每一行转换为json格式
            data = json.loads(line)
            
            # 防止字段不存在
            try:
                s = data["content"]
                buffer_data.append(s)
            except KeyError:
                print("注意，有字段不存在，跳过")
                continue
            
            # 计算字节数
            sum_bytes += len(buffer_data[-1].encode("utf-8"))
            
            if sum_bytes > file_bytes:
                # 将数据写入到新文件中
                with open(os.path.join(save_dir, f"{file_prefix}{int(count):04}{file_type}"), "w", encoding="utf-8") as f:
                    f.write('\n'.join(buffer_data))
                buffer_data = []
                sum_bytes = 0
                count += 1

# 记得最后剩余的数据也要写入新文件中
if sum_bytes != 0:
    with open(os.path.join(save_dir, f"{file_prefix}{int(count):04}{file_type}"), "w", encoding="utf-8") as f:
        f.write('\n'.join(buffer_data))
