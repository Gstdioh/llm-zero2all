"""
generate_instruction.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="gpt-3.5-turbo" \
"""
import time
import json
import os
import random
import regex as re
import string
from functools import partial
from multiprocessing import Pool
import unicodedata

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import jieba

import fire

import local_utils as utils


def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))


def tokenize(s):
    if contains_chinese(s):
        # 有中文，换一种分词方式
        return list(s.replace(" ", ""))  # 按字符分割
        # return jieba.lcut(s)  # 使用jieba分词
    else:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer._tokenizer.tokenize(s)


def encode_prompt(prompt_path, prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open(prompt_path).read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<|noinput|>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"<|{idx + 1}|>. Instruction: {instruction}\n"
        prompt += f"<|{idx + 1}|>. Input:\n{input}\n"
        prompt += f"<|{idx + 1}|>. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"<|{idx + 2}|>. Instruction:"
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"<|{num_prompt_instructions+1}|>. Instruction:" + response.message.content
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response.finish_reason == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"<\|{idx}\|>\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<|noinput|>" else input
            output = splitted_data[6].strip()
        
        if contains_chinese(inst):
            # 中文的话，按字符数来计算长度
            inst_len = len(list(inst.replace(" ", "")))
        else:
            # 英文的话，按单词数来计算长度
            inst_len = len(inst.split())
        # filter out too short or too long instructions
        if inst_len <= 3 or inst_len > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
            "图像",
            "图表",
            "图片",
            "文件",
            "地图",
            "绘制",
            "视频",
            "音频",
            "音乐",
            "流程图",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # 跳过第一个字符是标点符号的，包括中文标点符号
        if unicodedata.category(inst[0]).startswith('P'):
            continue
        # 非中文字符下，filter those starting with non-english character
        if not contains_chinese(inst) and not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./",
    prompt_path="./prompt.txt",
    seed_tasks_path="./seed_tasks.jsonl",
    save_path="regen.json",
    num_instructions_to_generate=10,
    model_name="gpt-3.5-turbo",
    num_prompt_instructions=3,
    request_batch_size=5,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    add_similar_info=True,
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    
    # 载入之前LLM生成的指令
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, save_path)):
        machine_instruction_data = utils.jload(os.path.join(output_dir, save_path))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # 进度
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # 目前所有的指令
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]
    
    # 分词，用于后面的相似度计算
    all_instruction_tokens = [tokenize(inst) for inst in all_instructions]

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        # 根据prompt构造指令的提示
        batch_inputs = []
        for _ in range(request_batch_size):
            # only sampling from the seed tasks
            prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
            prompt = encode_prompt(prompt_path, prompt_instructions)
            batch_inputs.append(prompt)
        stop_num = 20  # 每次生成的task数
        stop_num += 1
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=top_p,
            stop=[f"\n<|{stop_num}|>", f"<|{stop_num}|>. ", f"<|{stop_num}|>."],
        )
        
        # 使用open api来获取新的指令
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=model_name,
            batch_size=request_batch_size,
            decoding_args=decoding_args,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        # 进行一些后处理，将指令输入输出提取出来，并且过滤掉不合适的指令
        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
            instruction_data += new_instructions

        # 计算相似度，过滤掉相似度高的指令，保留相似度低的指令加入到machine_instruction_data中
        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:
            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = tokenize(instruction_data_entry["instruction"])
            with Pool(num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1
            
            # 是否添加相似度信息
            if add_similar_info:
                instruction_data_entry["most_similar_instructions"] = most_similar_instructions
                instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))

            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
            
            if len(machine_instruction_data) == num_instructions_to_generate:
                break
            
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
    
    save_path = os.path.join(output_dir, save_path)
    
    utils.jdump(machine_instruction_data, save_path)
    
    print(f"Save {len(machine_instruction_data)} machine_instruction_data to {save_path}")


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    # 记得添加环境变量
    # import os
    # os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
    # os.environ["OPENAI_BASE_URL"] = "your_openai_base_url"
    
    # 自动将 Python 对象（包括函数、类、对象、字典、列表等）转换为命令行接口
    # 传递一个函数（如 main）给 fire.Fire()，它会将命令行参数自动解析并传递给这个函数，类似argparse
    fire.Fire(main)