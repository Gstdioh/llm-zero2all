# 通过self-instruct构建指令数据集

from alpaca，构建命令如下：

```bash
python -m generate_instruction generate_instruction_following_data
```

文件描述
```txt
1. prompt.txt, 生成新指令的提示
2. seed_tasks.jsonl, 人工构造的种子指令集
```

构建流程：

1. 在seed_tasks.jsonl采样3个指令（包括输入输出）
2. 与prompt拼接
3. 将结果输入给gpt-3.5等模型（使用openai api）得到新的指令
4. 使用rouge_scorer._score_lcs计算新指令和种子指令的相似度，过滤掉最大相似度大于0.7的指令
5. 将过滤后的指令加入到种子指令集中
6. 重复1-5的操作，直到得到指定数量的新指令