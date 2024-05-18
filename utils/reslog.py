"""
在远程服务器上使用
"""

import os
import pickle


class ResLog:
    # result log
    def __init__(self, run_name="reslog", output_dir="./reslog", save_interval=1):
        self.run_name = run_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, run_name + ".pkl")
        self.save_interval = save_interval  # 保存为文件的间隔，即保存的频率
        
        self.log_data = {}
        
    def save(self, save_file_path):
        pickle.dump(self.log_data, open(save_file_path, "wb"))
        
    def load(self, load_file_path):
        self.log_data = pickle.load(open(load_file_path, "rb"))
        
    def log(self, data, name="0", step=None):
        """
        保存要记录的数据，每次log都要保存到文件中
        
        data:
            {
                "loss": 1,
                "acc": 0.2,
            }
        step: int, 记录的步数, None表明为当前长度，即如果一直设置为None，则结果为0, 1, 2, ...
        name: str, 记录的名称
        """
        # 若没有，则初始化
        if name not in self.log_data:
            self.log_data[name] = {}
            self.log_data[name]["step"] = []
        
        # 设置步数，即横坐标
        if step is None:
            self.log_data[name]["step"].append(len(self.log_data[name]["step"]))
        else:
            # 新设置的step必须要是严格递增的，如果不是，则此次不记录，退出
            if len(self.log_data[name]["step"]) > 0 and step <= self.log_data[name]["step"][-1]:
                return
            self.log_data[name]["step"].append(step)
            
        # 添加数据
        for key, value in data.items():
            if key not in self.log_data[name]:
                self.log_data[name][key] = []
            self.log_data[name][key].append(value)

        # 根据间隔来保存文件
        if len(self.log_data[name]["step"]) % self.save_interval == 0:
            # json.dump(self.log_data, open(self.log_file, "w"))  # 慢
            pickle.dump(self.log_data, open(self.log_file, "wb"))
