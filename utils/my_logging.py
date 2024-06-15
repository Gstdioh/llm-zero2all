import os
import sys
import logging
from logging import LogRecord, StreamHandler
import time
from datetime import datetime


BLACKLISTED_MODULES = ["torch.distributed", "parallel", "model"]


class CustomHandler(StreamHandler):
    """
    Custom StreamHandler to filter out logging from code outside of
    main code, and dump to stdout.
    """

    def __init__(self):
        super().__init__(stream=sys.stdout)
        console_formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s]: %(message)s')
        console_formatter.converter = time.localtime  # 使用本地时间
        self.setFormatter(console_formatter)

    def filter(self, record: LogRecord) -> bool:
        # Prevent log entries that come from the blacklisted modules
        # through (e.g., PyTorch Distributed).
        for blacklisted_module in BLACKLISTED_MODULES:
            if record.name.startswith(blacklisted_module):
                return False
        return True


class CustomFileHandler(logging.FileHandler):
    """
    Custom FileHandler to filter out logging from code outside of
    main code, and dump to stdout.
    """

    def __init__(self, filename, mode='a', encoding=None, delay=False, errors=None):
        super().__init__(filename, mode, encoding, delay, errors)
        console_formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s]: %(message)s')
        console_formatter.converter = time.localtime  # 使用本地时间
        self.setFormatter(console_formatter)

    def filter(self, record: LogRecord) -> bool:
        # Prevent log entries that come from the blacklisted modules
        # through (e.g., PyTorch Distributed).
        for blacklisted_module in BLACKLISTED_MODULES:
            if record.name.startswith(blacklisted_module):
                return False
        return True


def get_all_handlers(log_dir="./"):
    """
    获取所有的handler
    """
    custom_handler = CustomHandler()
    
    log_filename = f'info_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    custom_file_handler = CustomFileHandler(os.path.join(log_dir, log_filename))

    return [custom_handler, custom_file_handler]


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    """
    log_dir: 日志文件目录
    """
    log_filename = f'info_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    
    logger = logging.getLogger(name)  # 创建logger实例
    logger.setLevel(level)  # 设置级别

    # Add file handler and stdout handler，输出到文件中
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 特殊占位符，设置输出格式
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
