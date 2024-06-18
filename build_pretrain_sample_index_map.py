import tqdm
import logging

import numpy as np

import utils

logger = logging.getLogger(__name__)


def build_sample_index_map(file_list, max_seq_len, sample_index_map_path):
    progress_bar = tqdm.tqdm(total=len(file_list), desc=f"Build {sample_index_map_path}")

    sample_index_map = []  # [(file_start_id, sample_start_offset), ...]
    file_id = 0
    sample_start_offset = 0
    while file_id < len(file_list):
        file_path = file_list[file_id]
        try:
            file_m = np.memmap(file_path, dtype=np.uint16, mode="r")
        except:
            logger.warning(f"file {file_path} is empty, skipping")
            file_id += 1
            progress_bar.update(1)
            continue
        file_len = len(file_m) - sample_start_offset
        
        num_sample = file_len // max_seq_len
        remain_tokens = file_len % max_seq_len
        
        for i in range(num_sample):
            sample_index_map.append((file_id, sample_start_offset))
            sample_start_offset = sample_start_offset + max_seq_len

        if remain_tokens != 0:
            # 有剩余的token
            file_start_id = file_id
            file_id += 1
            progress_bar.update(1)
            while file_id < len(file_list):
                file_path = file_list[file_id]
                try:
                    file_m = np.memmap(file_path, dtype=np.uint16, mode="r")
                except:
                    logger.warning(f"file {file_path} is empty, skipping")
                    file_id += 1
                    progress_bar.update(1)
                    continue
                file_len = len(file_m)
                
                if file_len >= max_seq_len - remain_tokens:
                    # 该文件可以提供剩余的token
                    sample_index_map.append((file_start_id, sample_start_offset))
                    sample_start_offset = max_seq_len - remain_tokens
                    break
                else:
                    # 该文件不足以提供剩余的token，则继续找下一个文件
                    file_id += 1
                    remain_tokens += file_len
                    progress_bar.update(1)
        else:
            file_id += 1
            progress_bar.update(1)
       
    # 如果最后一个sample正好是一个文件的最后一个token，则不能作为样本，因为每个样本都需要保证后面还有一个token
    if remain_tokens == 0:
        sample_index_map = sample_index_map[:-1]

    # 转换为np.array
    sample_index_map = np.array(sample_index_map, dtype=np.uint32)  # 因为每个文件的token数不会超过2^32（100MB左右大小）

    # 保存np文件
    with open(sample_index_map_path, "wb") as f:
        f.write(sample_index_map.tobytes())
        
    num_samples = sample_index_map.shape[0]
    print(f"Build {sample_index_map_path} done, num_samples: {num_samples:,}")


max_seq_len = 2048
train_data_path = "data/02_train_data_more/01_bin_for_train_hf"
valid_data_path = "data/02_train_data_more/02_bin_for_valid_hf"

for data_path in [train_data_path, valid_data_path]:
    file_list = utils.get_file_paths(data_path, file_type="bin")

    build_sample_index_map(file_list, max_seq_len, f"{data_path}/all_sample_index_map_{max_seq_len}.ibin")
