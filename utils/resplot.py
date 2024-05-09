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
            
            print(f"row: {row}, col: {col}")
            
            idx = 0
            for key, value in data.items():
                for k, v in value.items():
                    if k == "step":
                        continue
                    plt.subplot(row, col, idx + 1)
                    plt.plot(value['step'], v, label=f"{key}.{k}")
                    plt.legend()
                    idx += 1
                idx += idx % col

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
        
    def get_remote_data(self, remote_file_path, local_file_path="reslog.pkl"):
        # 通过ssh来获取远程服务器的文件，并返回其中的数据
        
        # 创建SSH客户端
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # 连接到远程服务器
        ssh.connect(hostname=self.hostname, port=self.port, username=self.username, password=self.password)

        # 从Docker容器中获取文件
        sftp = ssh.open_sftp()
        sftp.get(remote_file_path, local_file_path)

        # 关闭连接
        sftp.close()
        ssh.close()

        # 在本地读取文件
        with open(local_file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data

    def run(self):
        i = 0
        while True:
            # self.reslog.log({
            #     "loss": 1 / (i + 1),
            #     "epoch": i,
            # })
            self.data = self.get_remote_data(self.remote_file_path)
            self.update_data.emit()  # 发送更新信号
            time.sleep(1)  # 线程暂停1秒
            i += 1
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="10.10.24.107")
    parser.add_argument("--port", type=int, default=30792)
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--password", type=str, default=r"32myp3M5fNwMXr^v5%7ubdLezPH2T0NE")
    parser.add_argument("--remote_file_path", type=str, default="/202232803052/axk/gly/llm-zero2all/utils/reslog.pkl")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    resplot = ResPlot(hostname=args.hostname, port=args.port, username=args.username, password=args.password,
                      remote_file_path=args.remote_file_path)
    resplot.show()
    
    sys.exit(app.exec_())