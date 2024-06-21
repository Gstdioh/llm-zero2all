import json
from utils import get_file_paths


find_text = "请使用计算机翻译系统，准确地将术语"
"""
这个rejected有点离谱

[{'from': 'human', 'value': '请使用计算机翻译系统，准确地将术语“人工智能”翻译成德语，同时考虑可能影响翻译过程的任何文化或语言细微差别。此外，您能否解释一下“人工智能”这个术语的起源和演变，以及它在现代技术中的重要性？'}]
'chosen': {'from': 'gpt', 'value': '人工智能的德语翻译是“Künstliche Intelligenz”。这个术语是由约翰·麦卡锡在1956年创造的，他组织了达特茅斯研讨会，这个研讨会通常被认为是人工智能研究的发源地。
'rejected': {'from': 'gpt', 'value': '机器学习\n(机器学习)'}}
"""


file_paths = get_file_paths("data/05_dpo_data/DPO-En-Zh-20k", file_type=["json"], startswith="", endswith="")

data_list = []

for file_path in file_paths:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for d in data:
            data_list.append(d)

for d in data_list:
    if find_text in str(d["conversations"]):
        print(d["content"])
        break
