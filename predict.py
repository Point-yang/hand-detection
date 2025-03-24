import socket
import json
import threading
import cv2
import time
from ultralytics import YOLO


class DetectionServer:
    def __init__(self):
        # 网络配置
        self.HOST = '0.0.0.0'  # 监听所有接口
        self.PORT = 65432
        self.MAX_CLIENTS = 5
        self.FRAME_RATE = 30

        # 摄像头配置
        self.CAMERA_ID = 0
        self.cap = None

        # 模型配置
        # predict.py
        self.model = YOLO("/app/model/best.pt")  # 使用容器内的相对路径
        self.class_names = ['激活', '静音', '确定', '音量增大' , '音量降低','暂停' ,'上一个' ,'返回']

        # 网络对象
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = False

    def init_camera(self):
        """初始化摄像头（带重试机制）"""
        max_retries = 3
        for i in range(max_retries):
            self.cap = cv2.VideoCapture(self.CAMERA_ID)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"摄像头 {self.CAMERA_ID} 初始化成功")
                return
            print(f"摄像头初始化失败，剩余重试次数: {max_retries - i - 1}")
            time.sleep(1)
        raise RuntimeError("无法打开摄像头")

    def generate_detections(self, frame):
        """生成检测结果（带异常保护）"""
        try:
            h, w = frame.shape[:2]
            results = self.model(frame, imgsz=640, conf=0.5, device=0)
            detections = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class": self.class_names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [
                        round(((x1 + x2) / 2) / w, 4),
                        round(((y1 + y2) / 2) / h, 4),
                        round((x2 - x1) / w, 4),
                        round((y2 - y1) / h, 4)
                    ]
                })
            return detections
        except Exception as e:
            print(f"检测错误: {str(e)}")
            return []

    def client_handler(self, conn, addr):
        """客户端线程（带稳定化处理）"""
        print(f"客户端连接: {addr}")
        try:
            while self.running:
                start_time = time.time()

                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("帧读取失败，重置摄像头...")
                    self.cap.release()
                    self.init_camera()
                    continue

                # 检测处理
                results = self.model(frame, imgsz=640, conf=0.5, device=0)  # 新增
                detections = self.generate_detections(frame)

                # ========== 新增可视化代码 ==========
                # 绘制检测框
                annotated_frame = results[0].plot(line_width=2)

                # 显示窗口
                cv2.imshow('Detection Preview', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                # ==================================

                # 构造数据包
                data = json.dumps({
                    "frame_id": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "detections": detections
                }).encode('utf-8')
                # 发送数据
                try:
                    header = len(data).to_bytes(4, 'big')
                    conn.sendall(header + data)
                except (BrokenPipeError, ConnectionResetError):
                    print(f"客户端 {addr} 断开连接")
                    break

                # 帧率控制
                elapsed = time.time() - start_time
                time.sleep(max(1 / self.FRAME_RATE - elapsed, 0))

        except Exception as e:
            print(f"客户端处理异常: {str(e)}")
        finally:
            conn.close()
            print(f"清理连接: {addr}")

    def start(self):
        """启动服务"""
        try:
            self.init_camera()
            self.server_socket.bind((self.HOST, self.PORT))
            self.server_socket.listen(self.MAX_CLIENTS)
            self.running = True
            print(f"服务端运行在 {self.get_ip()} : {self.PORT}")

            while self.running:
                conn, addr = self.server_socket.accept()
                thread = threading.Thread(target=self.client_handler, args=(conn, addr))
                thread.daemon = True
                thread.start()

        except Exception as e:
            print(f"服务启动失败: {str(e)}")
        finally:
            self.stop()

    def stop(self):
        """停止服务"""
        self.running = False
        if self.cap: self.cap.release()
        self.server_socket.close()
        print("服务已停止")

    def get_ip(self):
        """获取本机IP地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except:
            return "127.0.0.1"


if __name__ == "__main__":
    server = DetectionServer()
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()