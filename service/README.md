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

- `model_path`: `11mlabs/indri-0.1-124m-tts`
- `device`: `cuda:0`
- `port`: `8000`

Choices:

[HuggingFace collection](https://huggingface.co/collections/11mlabs/indri-673dd4210b4369037c736bfe)
