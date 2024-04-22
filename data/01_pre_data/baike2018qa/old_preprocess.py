import os
import glob

import json
from tqdm import tqdm

data_dir = "./"
file_prefix = "txt_baike_qa2018_"
# file_text_len = 2e6  # 大概5.1MB, *40=200MB
file_text_len = 80e6

# 获取当前文件夹下的所有文，除了valid数据集
filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
filenames = [filename for filename in filenames if not filename.endswith("valid.json")]

# 循环读取所有文件
i = 0
text = ""
count = 0
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as f:
        # 读取每一行
        lines = tqdm(f, desc='已完成0行')
        for line in lines:  #* 1. 先构建文件列表，再处理文件，一次性读取完文件，防止频繁IO
            # 将每一行转换为json格式
            data = json.loads(line)
            try:
                text += data["content"] + "\n"  #* 2. 将text改为列表，最后进行join，因为字符串拼接效率较低
            except KeyError:  # 有些数据中没有content字段
                continue
            i += 1
            
            if i % 1000 == 0:
                lines.set_description(f"已完成{i}行")
            
            if len(text) > file_text_len:  #* 3. 通过sum记录当前text列表中字符串的总长度，而不是每次都计算
                # 将数据写入到新文件中
                with open(os.path.join(data_dir, file_prefix + f"{int(count):04}.txt"), "w", encoding="utf-8") as ftxt:
                    ftxt.write(text)
                text = ""
                count += 1
