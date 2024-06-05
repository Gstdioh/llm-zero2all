"""
在本地上使用，可视化实验结果
"""

import sys
import time
import pickle
import argparse

import paramiko
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib import pyplot as plt


class ResPlot(QWidget):
    def __init__(self, **kwargs):
        super(ResPlot, self).__init__()
        self.kwargs = kwargs
        
        self.initData()
        
    def initData(self):
        self.resize(1300, 800)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 设置画布部分
        self.fig = plt.figure()
        self.myCanvas = FC(self.fig)
        # 设置布局，将组件添加到布局中
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.myCanvas)
        self.setLayout(self.layout)
        # 创建线程并连接信号
        self.myThread = MyThread(**self.kwargs)
        self.myThread.update_data.connect(self.updateData)
        self.myThread.start()
 
    def updateData(self):
        self.fig.clf()
        
        plt.title('实验结果', fontsize=12)
        
        data = self.myThread.data
        
        # 防止读取文件时远程服务器正好在写文件
        try:
            row = len(data.values())
            col = max([len(value) - 1 for value in data.values()])
            
            idx = 0
            for key, value in data.items():
                for k, v in value.items():
                    if k == "step":
                        continue
                    ax = plt.subplot(row, col, idx + 1)
                    label = f"{key}.{k}"
                    ax.plot(value['step'], v, label=label)
                    ax.legend()

                    # 获取横坐标和纵坐标的最后一个值
                    last_y = v[-1]

                    # 如果last_y是浮点数，那么保留4位小数，除了学习率
                    if isinstance(last_y, float):
                        if k == "mfu":
                            last_y = round(last_y, 2)
                        elif k == "lr":
                            last_y = round(last_y, 8)
                        else:
                            last_y = round(last_y, 4)
                    
                    # 在子图下面添加一个文本框，显示横坐标和纵坐标的最后一个值
                    ax.text(0.5, -0.2, f'{k}={last_y:,}', size=12, ha="center", transform=ax.transAxes)

                    idx += 1
                idx += idx % col

            # 增大子图之间的间距
            plt.subplots_adjust(hspace=0.5, wspace=0.5)

            self.myCanvas.draw()
        except:
            pass
    
 
class MyThread(QThread):
    update_data = pyqtSignal()

    def __init__(self, hostname, port, username, password, remote_file_path, parent=None):
        super(MyThread, self).__init__(parent)
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.remote_file_path = remote_file_path

        self.data = None

        # 创建SSH客户端
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # 连接到远程服务器
        self.ssh.connect(hostname=self.hostname, port=self.port, username=self.username, password=self.password)
        # 从Docker容器中获取文件
        self.sftp = self.ssh.open_sftp()

    def get_remote_data(self, remote_file_path, local_file_path="reslog.pkl"):
        # 如果读取文件时发生异常，那么程序会等待1秒再尝试读取文件
        while True:
            try:
                self.sftp.get(remote_file_path, local_file_path)

                # 在本地读取文件
                with open(local_file_path, 'rb') as f:
                    data = pickle.load(f)

                # 成功读取，则退出
                break
            except:
                time.sleep(1)  # 等待1秒再尝试读取文件

        return data

    def run(self):
        try:
            while True:
                self.data = self.get_remote_data(self.remote_file_path)
                self.update_data.emit()  # 发送更新信号
                time.sleep(3)  # 线程暂停3秒
        finally:
            # 保证关闭连接
            self.sftp.close()
            self.ssh.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="123.456.123.456")
    parser.add_argument("--port", type=int, default=9527)
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--password", type=str, default=r"123456")
    parser.add_argument("--remote_file_path", type=str, default="llm-zero2all/reslog/run2024_05_09_20_38_12.pkl")
    args = parser.parse_args()
    
    log_basename = args.remote_file_path.split("/")[-1]
    
    app = QApplication(sys.argv)
    app.setApplicationDisplayName(log_basename)
    
    resplot = ResPlot(hostname=args.hostname, port=args.port, username=args.username, password=args.password,
                      remote_file_path=args.remote_file_path)
    resplot.show()
    
    sys.exit(app.exec_())