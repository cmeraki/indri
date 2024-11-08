# Indri Inference Service

## Prerequisites

- Python 3.10+
- CUDA 12.1+

Install dependencies:

```bash
pip install -r service/requirements.txt
```

Install ffmpeg:

For linux:

```bash
sudo apt update -y
sudo apt upgrade -y
sudo apt install ffmpeg -y
```

## Running the service

```bash
python -m service.inference --model_path <model_path> --device <device> --port <port>
```

Defaults:

- `model_path`: `cmeraki/mimi_tts_hf`
- `device`: `cuda:0`
- `port`: `8000`

Choices:

- `model_path`: `cmeraki/mimi_tts_hf`, `cmeraki/mimi_tts_hf_stage`

## Directory Structure

- `inference.py`: Uvicorn server for the inference service.
- `models.py`: Pydantic models for the inference service.
- `tts.py`: vLLM model wrapper for the TTS model.
- `requirements.txt`: Python dependencies for the inference service.
