import sys


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
