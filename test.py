import socket
import json

class DetectionClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sock = None
        self.running = True

    def connect(self):
        """连接服务端（带超时机制）"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(5)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(None)  # 禁用超时
            print(f"成功连接到 {self.host}:{self.port}")
            return True
        except socket.timeout:
            print("连接超时")
            return False
        except Exception as e:
            print(f"连接错误: {str(e)}")
            return False

    def receive_data(self):
        """数据接收循环"""
        buffer = b''
        expected_length = 0

        try:
            while self.running:
                # 接收数据头
                if expected_length == 0:
                    header = self.sock.recv(4)
                    if not header: break
                    expected_length = int.from_bytes(header, byteorder='big')

                # 接收数据体
                while len(buffer) < expected_length:
                    chunk = self.sock.recv(expected_length - len(buffer))
                    if not chunk: break
                    buffer += chunk

                # 处理完整数据包
                if len(buffer) >= expected_length:
                    data = buffer[:expected_length]
                    buffer = buffer[expected_length:]
                    expected_length = 0

                    decoded = json.loads(data.decode('utf-8'))
                    self.print_detection(decoded)

        except Exception as e:
            print(f"接收错误: {str(e)}")
        finally:
            self.sock.close()

    def print_detection(self, data):
        """格式化打印结果"""
        print(f"\n帧号: {data['frame_id']}")
        print(f"检测到 {len(data['detections'])} 个目标:")
        for det in data['detections']:
            bbox = det['bbox']
            print(f" - {det['class']}: {det['confidence'] * 100:.1f}%")
            print(f"   中心点: ({bbox[0]:.3f}, {bbox[1]:.3f})")
            print(f"   尺寸: {bbox[2]:.3f}x{bbox[3]:.3f}")
        print("─" * 50)

    def run(self):
        if self.connect():
            self.receive_data()


if __name__ == "__main__":
    # 修改为实际服务端IP
    client = DetectionClient("192.168.2.235", 65432)
    client.run()