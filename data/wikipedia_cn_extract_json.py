import os
from gensim.corpora.wikicorpus import extract_pages, filter_wiki
import bz2file
import re
from opencc import OpenCC
from tqdm import tqdm
import codecs

import pandas as pd
import jsonlines
import json

cc = OpenCC('t2s')

data_dir = "./wikipedia_cn_20240418"
file_prefix = "wikipedia_cn_20240416_"
# file_text_len = 1e6  # 中文json，大概2.5MB, *80=100MB
file_text_len = 90e6

os.makedirs(data_dir, exist_ok=True)

def wiki_replace(d):
    global cc
    s = d[1]
    s = re.sub(r':*{\|[\s\S]*?\|}', '', s)
    s = re.sub(r'<gallery>[\s\S]*?</gallery>', '', s)
    s = re.sub(r'(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
    s = filter_wiki(s)
    s = re.sub(r'\* *\n|\'{2,}', '', s)
    s = re.sub(r'\n+', '\n', s)
    s = re.sub(r'\n[:;]|\n +', '\n', s)
    s = re.sub(r'\n==', '\n\n==', s)
    return cc.convert(s).strip()


# 提取文章
wiki = extract_pages(bz2file.open(
    '/home/guoliuyang/code/03_LLM/llm-zero2all/data/zhwiki-latest-pages-articles-multistream.xml.bz2'))
i = 0
# f = codecs.open('wiki.cn.txt', 'w', encoding='utf-8')
w = tqdm(wiki, desc='已获取0篇文章')

json_lines = []
size_sum = 0
count = 0
for d in w:  # Title, text, and page id
    # re.findall('^[a-zA-Z]+:', d[0]) is used to remove help pages
    # re.findall(u'^#', d[1]) is used to remove redirect pages
    if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall('^#', d[1]):
        s = wiki_replace(d)
        i += 1
        if i % 100 == 0:
            w.set_description('已获取%s篇文章' % i)
        
        data = {
            "id": i-1,
            "title": cc.convert(d[0]).strip(),
            "text": s,
        }
        json_lines.append(json.dumps(data, ensure_ascii=False))  # 注意加上ensure_ascii=False，否则会保存为Unicode编码
        size_sum += len(json_lines[-1])
        
        if size_sum > file_text_len:
            with open(os.path.join(data_dir, file_prefix + f"{int(count):04}.txt"), "w", encoding="utf-8") as f:
                f.write('\n'.join(json_lines))
            json_lines = []
            size_sum = 0
            count += 1
