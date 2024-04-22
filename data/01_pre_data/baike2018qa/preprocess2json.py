import os
import glob
import regex as re
import unicodedata

import json
from tqdm import tqdm

"""
json文件
{
    gid: int, id: int, data_src: str, category: str, title: str, content: str, desc: str,
    others: dict
}
"""

# 未使用
def is_special_char(c):
    # Unicode范围对应特殊符号和图形
    special_char_ranges = [
        '\u2600-\u26FF',  # 杂项符号
        '\u2190-\u21FF',  # 箭头
        '\u2200-\u22FF',  # 数学运算符
        '\u25A0-\u25FF',  # 几何形状
        '\u2300-\u23FF',  # 杂项技术
        '\u2500-\u257F',  # 箱图元素
        '\u2580-\u259F'   # 块元素
    ]

    if c >= '\u2600' and c <= '\u26FF' or c >= '\u2190' and c <= '\u21FF' or c >= '\u2200' and c <= '\u22FF' or c >= '\u25A0' and c <= '\u25FF' or c >= '\u2300' and c <= '\u23FF' or c >= '\u2500' and c <= '\u257F' or c >= '\u2580' and c <= '\u259F':
        return True
    return False

# 注意：这里的清洗方法只是简单的清洗，不一定适用于所有的数据集
# 对中文使用0, 1, 2, 3, 4
# 对英文使用0, 1, 2, 3, 4
# 对code使用4
def clean_text(s, type="zh"):
    # 只保留"\n"和所有可显示的字符
    # s = ''.join(c for c in s if unicodedata.category(c) not in ["Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"] or c == '\n')
    # s = s.strip()  # 去除首尾空格
    # s = unicodedata.normalize('NFKC', s)  # 规范化，将全角字符转换为半角字符，弃用：在tokenizer中进行规范化；或者不进行规范化
    
    # 新的清洗方法
    if type == "zh" or type == "en":
        s = ''.join(c for c in s if unicodedata.category(c) not in ["Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"] or c == '\n')  # 0. 只保留"\n"和所有可显示的字符
        s = re.sub(r'([。？！，、；：“”’‘（）——…《》【】·\!\\"\#\$\%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~])\1+', r'\1', s)  # 1. 去除重复的标点符号
        s = re.sub(r'\s+([\u4e00-\u9fa5。？！，、；：“”’‘（）——…《》【】·\!\\"\#\$\%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~])', r'\1', s)  # 2. 去除符号（包括中文字符）前的空格，这样能去除符号之间的空格
        s = re.sub(r'\s+([\S])', r' \1', s)  # 3. 只保留一个空格
    s = s.strip()  # 4. 去除首尾空格
    
    return s

data_src = "baike2018qa"
file_type = ".json"

data_dir = "./raw_baike2018qa"
save_dir = f"./json_{data_src}"
file_prefix = f"json_{data_src}_"
file_bytes = 200 * 1024 * 1024
os.makedirs(save_dir, exist_ok=True)

# 获取当前文件夹下的所有文件，除了valid数据集
filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
filenames = [filename for filename in filenames if not filename.endswith("valid.json")]

# 循环读取所有文件
buffer_data = []  # 一个文件的数据
sum_bytes = 0  # 
count = 0
gid = 0
id = 0
special_char_count = 0
for filename in filenames:
    with open(filename, "r", encoding="utf-8") as fsrc:
        # 读取所有行
        lines = list(tqdm(fsrc, desc=f'构建"{filename}"文件列表'))
        lines = tqdm(lines, desc=f'处理"{filename}"')
        for line in lines:
            # 将每一行转换为json格式
            data = json.loads(line)
            
            # 跳过特殊字符
            if "▅" in data["answer"] or '█' in data["answer"] or '▂' in data["answer"]:
                print(f"跳过包含特殊符号▅█▂的文本")
                special_char_count += 1
                continue
            
            # 跳过字段不存在的数据
            try:
                new_data = {
                    # 通用
                    "gid": gid,
                    "id": id,
                    "data_src": data_src,
                    "category": clean_text(data["category"]),
                    "title": clean_text(data["title"]),
                    "content": clean_text(data["answer"]),
                    "desc": clean_text(data["desc"]),
                    # 其他
                    "others": {
                        "qid": data["qid"]
                    }
                }
                gid += 1
                id += 1
                buffer_data.append(json.dumps(new_data, ensure_ascii=False))
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
                id = 0

# 记得最后剩余的数据也要写入新文件中
if sum_bytes != 0:
    with open(os.path.join(save_dir, f"{file_prefix}{int(count):04}{file_type}"), "w", encoding="utf-8") as f:
        f.write('\n'.join(buffer_data))

with open(f"./{data_dir}_special_char_count_new.txt", "w", encoding="utf-8") as f:
    f.write(f"{data_dir}中的特殊符号的数量为：{special_char_count}个\n")
