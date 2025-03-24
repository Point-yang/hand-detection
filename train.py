from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

# 使用余弦退火学习率调度进行训练
results = model.train(
    data='hand.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    workers=0,
    lr0=0.001,  # 初始学习率
    lrf=0.01,  # 最终学习率因子，这里是初始学习率的 1%
    lrs='cos',  # 指定使用余弦退火学习率调度
    batch=8,
    amp=True,
)
