# handsdetection示例

本项目演示了通过mediapipe提取手部后基于yolov8训练手势识别模型，并通过socket协议传输识别结果。



## 系统要求

· Python 3.12.9

· 依赖项 （通过安装要求自动安装）：

​           · opencv-python

​           · mediapipe







### 安装依赖：

```bash
pip install -r requirements.txt
```

初次使用，打开hand.yaml修改各文件对应的路径

1.打开caiyang.py，按1~8分别采集不同的手势，采集完成按Esc退出。（保存在dataset_raw文件下）

2.运行labeling.py

3.打开data_divide.py，修改根目录路径为PythonProject2的绝对路径后运行

4.运行train.py

5.打开predict.py，修改训练的模型路径，运行

等待客户端连接
