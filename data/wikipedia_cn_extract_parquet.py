
import os
from gensim.corpora.wikicorpus import extract_pages, filter_wiki
import bz2file
import re
from opencc import OpenCC
from tqdm import tqdm
import codecs

import pandas as pd

cc = OpenCC('t2s')


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

# Create a DataFrame to store the processed text, title, and id
df = pd.DataFrame(columns=['id', 'title', 'text'])
for d in w:  # Title, text, and page id
    # re.findall('^[a-zA-Z]+:', d[0]) is used to remove help pages
    # re.findall(u'^#', d[1]) is used to remove redirect pages
    if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall('^#', d[1]):
        s = wiki_replace(d)
        df = df._append({'id': i, 'title': cc.convert(d[0]).strip(), 'text': s}, ignore_index=True)
        i += 1
        if i % 100 == 0:
            w.set_description('已获取%s篇文章' % i)
        size = 5e4
        if i % size == 0:
            # Save the DataFrame as a Parquet file
            folder_path = './wikipedia_cn_20240416_my'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            df.to_parquet(f'./wikipedia_cn_20240416_my/wiki_cn_20240416_my_{int(i/size-1):03}.parquet')
            df = pd.DataFrame(columns=['id', 'title', 'text'])
