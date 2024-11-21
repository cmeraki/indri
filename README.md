# Indri

[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-11mlabs-blue)](https://huggingface.co/11mlabs)

Multimodal audio LMs for TTS, ASR, and voice cloning

## Running locally

### Prerequisites

- Python 3.10+
- CUDA 12.1+

Install dependencies:

```bash
pip install -r requirements.txt
```

Install [ffmpeg](https://www.ffmpeg.org/download.html):

For linux:

```bash
sudo apt update -y
sudo apt upgrade -y
sudo apt install ffmpeg -y
```

### Running the service

```bash
python -m inference --model_path 11mlabs/indri-0.1-124m-tts --device cuda:0 --port 8000
```

Defaults:

- `device`: `cuda:0`
- `port`: `8000`

Choices:

- `model_path`: [HuggingFace collection](https://huggingface.co/collections/11mlabs/indri-673dd4210b4369037c736bfe)

Redirect to `http://localhost:8000/docs` to see the API documentation and test the service.
