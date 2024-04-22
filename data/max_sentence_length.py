import os
import glob

from tqdm import tqdm

dirs = ['baike2018qa', 'github-python', 'news2016zh', 'webtext2019zh', 'wikipedia_cn_20240418', 'wikipedia_en_20220301']

max_len = 0
count = 0
lines = 0
size = 10000

for dir in dirs:
    file_list = sorted(glob.glob(f'./{dir}/txt_{dir}/*'))
    for filename in tqdm(file_list):
        with open(filename, 'r', encoding='utf-8') as f:
            while (line := f.readline()):
                max_len = max(max_len, len(line.encode('utf-8')))
                if len(line.encode('utf-8')) > size:
                    count += 1
                lines += 1
    print("---------------------------------------------------")
    print(dir)
    print(max_len)
    print(count)
    print(lines)

print("---------------------------------------------------")
print(max_len)
print(count)
print(lines)