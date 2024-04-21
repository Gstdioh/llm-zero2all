import os
import random

import json
from tqdm import tqdm

# 注意，进行了随机打乱


# 获取当前文件夹下的所有文件夹
def get_dir_abspaths(file_dir: str) -> list:
    dir_abspaths = sorted([os.path.join(file_dir, i) for i in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, i))])
    return dir_abspaths


# 获取当前文件夹下的所有文件
def get_file_abspaths(file_dir: str) -> list:
    file_abspaths = sorted([os.path.join(file_dir, i) for i in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, i))])
    return file_abspaths


file_type = "txt"
save_dir = "./00_txt_for_train_tokenizer"

os.makedirs(save_dir, exist_ok=False)

data_dir = "./"
dir_list = get_dir_abspaths(data_dir)

# 获取所有文件
all_file_list = []
for dir_path in dir_list:
    # 获取所有文件
    file_list = get_file_abspaths(dir_path)
    
    all_file_list += file_list

random.seed(42)
random.shuffle(all_file_list)

# 循环处理所有文件
file_bytes = 200 * 1024 * 1024  # 一个txt文件200MB
buffer_data = []  # 一个文件的数据
sum_bytes = 0
count = 0
for file_path in tqdm(all_file_list):
    with open(file_path, "r", encoding="utf-8") as fjson:
        # 读取所有行
        for line in fjson.readlines():
            # 将每一行转换为json格式
            data = json.loads(line)
            
            # 防止字段不存在
            try:
                # 按照不同情况，添加额外的标点符号
                if len(data["title"] + data["desc"]) == 0:
                    s = data["content"]
                else:
                    s = data["title"] + data["desc"] + "\n" + data["content"]
                buffer_data.append(s)
            except KeyError:
                print("注意，有字段不存在，跳过")
                continue
            
            # 计算字节数
            sum_bytes += len(buffer_data[-1].encode("utf-8"))
            
            if sum_bytes > file_bytes:
                # 将数据写入到新文件中
                with open(os.path.join(save_dir, f"{file_type}_for_train_tokenizer_{int(count):04}.{file_type}"), "w", encoding="utf-8") as f:
                    f.write('\n'.join(buffer_data))
                buffer_data = []
                sum_bytes = 0
                count += 1

# 记得写入剩余的数据
if sum_bytes > 0:
    with open(os.path.join(save_dir, f"{file_type}_for_train_tokenizer_{int(count):04}.{file_type}"), "w", encoding="utf-8") as f:
        f.write('\n'.join(buffer_data))
