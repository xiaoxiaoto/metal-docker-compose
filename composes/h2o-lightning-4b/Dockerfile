FROM python:3.12-slim

# 官方 BOX tightness / hybrid 路径绘框需要 Pillow；模型仍只在宿主 vLLM Metal 运行。
RUN python -m pip install --no-cache-dir --no-deps Pillow==13.0.0
