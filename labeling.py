import os
import cv2
import mediapipe as mp
import shutil

# 配置参数
SAVE_ROOT = "dataset_raw"  # 数据保存根目录
CLASSES = ["gesture_0", "gesture_1", "gesture_2", "gesture_3", "gesture_4", "gesture_5", "gesture_6",
           "gesture_7"]  # 手势类别目录
CLASS_ID_MAP = {
    "gesture_0": 0,
    "gesture_1": 1,
    "gesture_2": 2,
    "gesture_3": 3,
    "gesture_4": 4,
    "gesture_5": 5,
    "gesture_6": 6,
    "gesture_7": 7
}  # 类别与 ID 的映射
LABELS_DIR = "label"  # 标签文件夹
IMAGE_DIR = "image"  # 图片文件夹

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)


def create_yolo_annotation(image_path, hand_bounding_box, class_id):
    """
    创建 YOLO 格式的标签文件
    :param image_path: 图像路径
    :param hand_bounding_box: 手部边界框 [xmin, ymin, xmax, ymax]
    :param class_id: 类别 ID
    """
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    x_min, y_min, x_max, y_max = hand_bounding_box
    x_center = (x_min + x_max) / 2.0 / width
    y_center = (y_min + y_max) / 2.0 / height
    bbox_width = (x_max - x_min) / width
    bbox_height = (y_max - y_min) / height

    label_line = f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n"

    # 生成标签文件路径
    base_name = os.path.basename(image_path)
    txt_file = os.path.join(LABELS_DIR, os.path.splitext(base_name)[0] + ".txt")

    # 写入标签文件
    with open(txt_file, 'w') as f:
        f.write(label_line)


def detect_and_label_hands():
    # 确保标签文件夹和图片文件夹存在
    if not os.path.exists(LABELS_DIR):
        os.makedirs(LABELS_DIR)
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    for cls in CLASSES:
        class_dir = os.path.join(SAVE_ROOT, cls)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist.")
            continue

        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image in images:
            image_path = os.path.join(class_dir, image)
            image_data = cv2.imread(image_path)
            results = hands.process(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * image_data.shape[1]), int(landmark.y * image_data.shape[0])
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    # 获取类别 ID
                    class_id = CLASS_ID_MAP[cls]

                    # 创建 YOLO 标注文件
                    create_yolo_annotation(image_path, [x_min, y_min, x_max, y_max], class_id)
                    print(f"Generated YOLO label for: {image_path}")

                    # 将图片移动到 IMAGE_DIR 文件夹
                    new_image_path = os.path.join(IMAGE_DIR, image)
                    shutil.move(image_path, new_image_path)
                    print(f"Moved image to: {new_image_path}")
            else:
                # 如果没有检测到手部，则删除该图片
                os.remove(image_path)
                print(f"No hand detected for: {image_path}, image deleted.")


if __name__ == "__main__":
    detect_and_label_hands()
