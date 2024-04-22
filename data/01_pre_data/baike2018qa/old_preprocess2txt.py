import os
import glob
import regex as re
import unicodedata

import json
from tqdm import tqdm

def clean_text(s):
    # 保留"\n"和所有可显示的字符
    s = ''.join(c for c in s if unicodedata.category(c) not in ["Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"] or c == '\n')
    s = s.strip()  # 去除首尾空格
    # s = unicodedata.normalize('NFKC', s)  # 规范化，将全角字符转换为半角字符，弃用：在tokenizer中进行规范化；或者不进行规范化
    return s

data_dir = "./"
save_dir = "./txt_baike2018qa"
file_prefix = "txt_baike2018qa_"
file_text_len = 200 * 1024 * 1024
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
                # 按照不同情况，添加额外的标点符号
                tmp = ""
                if len(data["desc"]) > 0:
                    tmp = '，' + clean_text(data["desc"]) + '。'
                s = clean_text(data["title"]) + tmp + clean_text(data["answer"]) + '\n'
                texts.append(s)
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
