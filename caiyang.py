# -*- coding: utf-8 -*-
import cv2
import os
from datetime import datetime

# 配置参数
SAVE_ROOT = "dataset_raw"  # 数据保存根目录
CLASSES = ["gesture_0","gesture_1","gesture_2","gesture_3","gesture_4", "gesture_5", "gesture_6","gesture_7"]  # 手势类别目录
IMAGE_QUALITY = 95  # 保存图片质量 (0-100)
AUTO_SAVE_INTERVAL = 0.5  # 自动保存间隔(秒)
RESOLUTION = (1280, 720)  # 摄像头分辨率

# 初始化摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

# 创建保存目录
os.makedirs(SAVE_ROOT, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(SAVE_ROOT, cls), exist_ok=True)

# 状态跟踪变量
last_save_time = 0
counter = {cls: 0 for cls in CLASSES}  # 各类别计数器

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # 显示实时画面
    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 1-8 to save | ESC to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示帮助信息
    for idx, cls in enumerate(CLASSES, 1):
        cv2.putText(display_frame, f"{idx}: {cls} ({counter[cls]})",
                    (10, 60 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow("Data Collection", display_frame)

    # 按键处理
    key = cv2.waitKey(10)

    # ESC退出
    if key == 27:
        break

    # 数字键1-3保存对应类别
    if 49 <= key <= 56:  # ASCII码 1-8
        cls_idx = key - 49
        cls_name = CLASSES[cls_idx]

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{cls_name}_{timestamp}.jpg"
        save_path = os.path.join(SAVE_ROOT, cls_name, filename)

        # 保存原始图像
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        print(f"Saved: {save_path}")
        counter[cls_name] += 1

        # 显示保存反馈
        cv2.putText(display_frame, f"Saved to {cls_name}!", (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("Data Collection", display_frame)
        cv2.waitKey(300)  # 显示反馈信息300ms

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")
print("Final counts:", counter)