# docker run --gpus all indri

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV MODEL_PATH=cmeraki/mimi_tts_hf
ENV DEVICE=cuda:0
ENV PORT=8000

EXPOSE 8000

COPY service/requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
COPY service /app

ENTRYPOINT ["python3", "-m", "inference", "--model_path", "${MODEL_PATH}", "--device", "${DEVICE}", "--port", "${PORT}"]
