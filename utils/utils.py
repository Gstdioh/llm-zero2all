import os
import sys
import json
import regex as re
import unicodedata
import subprocess


# 主进程才会输出信息
_ddp = int(os.environ.get("RANK", -1)) != -1
_master_process = True
_ddp_rank = 0
_ddp_local_rank = 0
_ddp_world_size = 1
if _ddp:
    _ddp_rank = int(os.environ["RANK"])
    _master_process = _ddp_rank == 0
    _ddp_local_rank = int(os.environ["LOCAL_RANK"])
    _ddp_world_size = int(os.environ["WORLD_SIZE"])
    
    
def print_rank0(print_fn, message=None):
    """
    只在主进程打印信息
    """
    _ = print_fn(message) if _master_process else None


def is_master_process():
    """
    是否是主进程
    """
    return _master_process


def is_local_rank0():
    """
    是否是本地的第一个进程
    """
    return _ddp_local_rank == 0


def is_ddp():
    """
    是否是分布式训练
    """
    return _ddp


# 解析命令行参数
def kwargs_parse():
    kwargs = {}
    args = sys.argv[1:]  # 去除第一个参数，即文件名
    i = 0
    
    # 解析help
    if i < len(args) and (args[0] == "-h" or args[0] == "--help"):
        kwargs["help"] = True
        i = len(args)
    
    while i < len(args):
        if args[i].startswith("--"):
            arg_name = args[i][2:]  # 去除"--"前缀
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                arg_value = args[i + 1]
                i += 1
            else:
                arg_value = True  # 若没有变量值，则默认为True
            if arg_value.isdigit():  # 检查是否是整数
                arg_value = int(arg_value)
            elif arg_value.replace(".", "").isdigit():  # 检查是否是浮点数
                arg_value = float(arg_value)
            kwargs[arg_name] = arg_value
        i += 1
        
    return kwargs


def convert_mem2num(mem_str: str) -> int:
    '''
    将内存字符串转换为数字
    如 "2M" -> 2 * 1024 * 1024
    '''
    # 转为大写
    mem_str = mem_str.upper()
    if mem_str[-1] == "K":
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str[-1] == "M":
        return int(float(mem_str[:-1]) * 1024 * 1024)
    elif mem_str[-1] == "G":
        return int(float(mem_str[:-1]) * 1024 * 1024 * 1024)
    else:
        raise ValueError("内存字符串格式错误！单位应为K、M、G！")


# 构建迭代器来训练，防止OOM
# 经过训练，在22G左右的数据下，使用迭代器训练一样会卡住，在进度57/185的时候，占用89%内存，然后就卡住了
def get_training_iterator(files_for_train_tokenizer: list, buffer_bytes="2M", max_train_bytes="5G"):
    # 类似data/02_train_data/get_txt_for_tokenizer.py中的处理
    # buffer_bytes: 每次生成buffer_bytes的数据
    buffer_bytes = convert_mem2num(buffer_bytes)
    max_train_bytes = convert_mem2num(max_train_bytes)
    
    # 循环处理所有文件
    buffer_data = []  # 一个文件的数据
    sum_bytes = 0
    train_bytes = 0
    for file_path in files_for_train_tokenizer:
        with open(file_path, "r", encoding="utf-8") as fjson:
            # 读取所有行
            for line in fjson.readlines():
                # 将每一行转换为json格式
                data = json.loads(line)
                
                # 按照不同情况，添加额外的标点符号
                if len(data["title"] + data["desc"]) == 0:
                    s = data["content"]
                else:
                    s = data["title"] + data["desc"] + "\n" + data["content"]
                buffer_data.append(s)
                
                # 计算字节数
                tmp = len(buffer_data[-1].encode("utf-8"))
                sum_bytes += tmp
                train_bytes += tmp
                
                if sum_bytes > buffer_bytes:
                    yield buffer_data
                    buffer_data = []
                    sum_bytes = 0
                    
                    if train_bytes > max_train_bytes:
                        break
                    
        if train_bytes > max_train_bytes:
            break
                    
    # 记得最后剩余的数据
    if len(buffer_data) > 0:
        yield buffer_data


def get_file_paths(file_dir: str, file_type=["json"], startswith: str = "", endswith: str = "") -> list:
    '''
    获取当前文件夹下某种类型的所有文件的绝对路径，要递归遍历所有子文件夹
    
    若file_type is None, 则返回所有文件
    '''
    # 如果给的是文件，则直接返回文件列表
    if os.path.isfile(file_dir):
        return [file_dir]
    
    if isinstance(file_type, str):
        file_type = [file_type]
    
    file_abspaths = []
                
    def dfs_get_file_paths(file_dir: str):
        for file in sorted(os.listdir(file_dir)):
            file_path = os.path.join(file_dir, file)
            if os.path.isdir(file_path):
                dfs_get_file_paths(file_path)
            elif file_type is None or (file.split(".")[-1] in file_type and file.startswith(startswith) and file.endswith(endswith)):
                file_abspaths.append(file_path)
                
    dfs_get_file_paths(file_dir)

    return file_abspaths


def find_file_path(file_dir, file_basename, allow_multi=False):
    """"
    根据文件基本名，在目录中找到该文件的绝对路径
    
    allow_multi=False下，若找到多个文件，则报错，默认不允许多个文件
    """
    # 首先找到file_dir下的所有文件
    file_paths = get_file_paths(file_dir, file_type=None)
    
    found_file_paths = []
    for file_path in file_paths:
        if os.path.basename(file_path) == file_basename:
            found_file_paths.append(file_path)
            if not allow_multi and len(found_file_paths) > 1:
                raise ValueError(f"找到多个文件：{found_file_paths}，请检查！")
            
    return found_file_paths

def get_file_line_count(file_path):
    """
    通过命令行快速获取文件的行数
    """
    result = subprocess.run(["wc", "-l", file_path], stdout=subprocess.PIPE)
    return int(result.stdout.split()[0])


def is_json_file(file_path):
    """
    判断是否是json文件，主要用来和jsonl区分
    
    通过简单判断，后缀为.json且第一个字符为[，则认为是json文件
    """
    with open(file_path, "r") as f:
        if f.read(1) == "[" and file_path.endswith(".json"):
            return True
    return False


def split_list_avg(data_list, split_num):
    """
    将列表平均分为split_num份
    """
    block_size = len(data_list) // split_num
    
    if block_size == 0:
        return data_list
    
    ret_list = []
    for block_id in range(0, split_num - 1):
        ret_list.append(data_list[block_id * block_size : (block_id + 1) * block_size])
    # 最后一个block取所有数据
    ret_list.append(data_list[(split_num - 1) * block_size:])

    return ret_list


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


def save_run_exp_config_deprecated(save_path, exp_config=None, run_code_abspath=None):
    '''
    弃用，因为run_code_abspath获取的不一定是值，可能是类似下面这样的，有误
    out_dir = os.path.join(out_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    
    可以使用两种方式获取当前实验参数配置
    
    1. exp_config：传入当前实验的配置信息字典
    2. run_code_abspath：传入当前运行的代码的绝对路径，会自动解析出当前实验的配置信息（推荐，因为会将注释也保存了）
    
    保存当前运行实验的配置信息，写入python文件中
    找到下面间隔的代码
    
    # -----------------------------------------------------------------------------
    '''
    if exp_config is not None and run_code_abspath is not None:
        raise ValueError("exp_config和run_code_abspath不能同时传入！")
    if exp_config is None and run_code_abspath is None:
        raise ValueError("exp_config和run_code_abspath不能同时为空！")
    
    exp_config_text = "# 最终的配置文件信息\n"
    if exp_config is not None:
        for key, value in exp_config.items():
            if isinstance(value, str):
                exp_config_text += f"{key} = \"{value}\"\n"
            else:
                exp_config_text += f"{key} = {value}\n"
    elif run_code_abspath is not None:
        with open(run_code_abspath, "r", encoding="utf-8") as f:
            code_text = f.read()
        split_str = "# -----------------------------------------------------------------------------"
        code_text_list = code_text.split(split_str)
        
        # 将分割的第二段内容（即实验参数配置），保存到exp_config_text中
        exp_config_text += code_text_list[1].strip()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(exp_config_text)


def save_run_exp_config(save_path, exp_config):
    '''
    保存当前运行实验的配置信息，写入python文件中
    找到下面间隔的代码
    # -----------------------------------------------------------------------------
    '''
    exp_config_text = "# 最终的配置文件信息\n"
    
    for key, value in exp_config.items():
        if isinstance(value, str):
            exp_config_text += f"{key} = \"{value}\"\n"
        else:
            exp_config_text += f"{key} = {value}\n"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(exp_config_text)


def unwrap_model(model):
    """
    递归地将模型的嵌套模型解开，返回一个模型列表
    """
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model
