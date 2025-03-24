# Dockerfile
FROM python:3.12-bookworm

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制模型文件和代码
COPY requirements.txt .
COPY . .
COPY runs/detect/train8/weights/best.pt ./model/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["python", "predict.py"]