"""
Copy from llama.c/configurator.py

Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import os
import sys
from ast import literal_eval

from utils import print_rank0


for arg in sys.argv[1:]:  # 对于python main.py "123 123" sys.argv 结果是 ['main.py', '123 123'] 双引号被视为一个整体
    # 可以--resume=True或者--resume
    # 注意=两边不能有空格
    if arg.startswith('--'):
        if "=" not in arg:
            # --resume, means resume=True
            key = arg[2:]
            if key in globals():
                print_rank0(print, f"Overriding: {key} = True")
                globals()[key] = True
            else:
                raise ValueError(f"Unknown config key: {key}")
        else:
            # assume it's a --key=value argument
            key, val = arg.split('=')
            key = key[2:]
            if key in globals():
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)  # 安全地解析一个字符串为一个Python字面量或表达式
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok，不用进行确定
                # assert type(attempt) == type(globals()[key]), f"Type mismatch! {key} = {attempt} vs {globals()[key]}"
                # cross fingers
                print_rank0(print, f"Overriding: {key} = {attempt}")
                globals()[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")
