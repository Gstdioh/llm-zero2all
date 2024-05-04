import os
import sys
import logging
import time


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    """
    log_dir: 日志文件目录
    """
    
    logger = logging.getLogger(name)  # 创建logger实例
    logger.setLevel(level)  # 设置级别

    # Add file handler and stdout handler，输出到文件中
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # 特殊占位符，设置输出格式
    formatter.converter = time.localtime  # 使用本地时间
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)

    # Add console handler，输出到控制台
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter.converter = time.localtime  # 使用本地时间
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    # 添加处理器到logger中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)

    return logger
