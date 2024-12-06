# Indri GGUF Inference

Refer to the original model and more details [here](https://huggingface.co/11mlabs/indri-0.1-124m-tts).

This guide will help in running Indri models on CPU in GGUF format.

## Step 1: Build llama.cpp

To run the inference locally, you need to build `llama.cpp` project. The updated guide to do so can be found [here](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md).

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

cmake -B build
cmake --build build --config Release
```

## Step 2: Download the model

Download the GGUF format models from HuggingFace and place them inside `llama.cpp/models/`.
The models can be found [here](https://huggingface.co/11mlabs/indri-0.1-124m-tts-GGUF/tree/main).

Once the model is placed inside the directory, run the `llama-cpp` server from inside the `llama.cpp` directory

```bash
# For F16 model, update for different quantization accordingly
./build/bin/llama-server -m /indri-0.1-124M-tts-F16.gguf --samplers 'top_k;temperature' --top_k 15
```

Refer [here](https://github.com/ggerganov/llama.cpp/tree/master/examples/main) if you are facing issues in running the llama-server locally.

## Step 3: Run the inference script

Clone the GitHub repository:

```bash
git clone https://github.com/cmeraki/indri.git
cd indri

python -m src.tts_gguf --text 'hi my name is Indri' --speaker '[spkr_63]' --out out.wav
```

Speakers are available [here](https://github.com/cmeraki/indri?tab=readme-ov-file#available-speakers).

You can also run an inference server

```bash
pip install -r requirements.txt

# Install ffmpeg (for Mac/Windows, refer here: https://www.ffmpeg.org/download.html)
sudo apt update -y
sudo apt upgrade -y
sudo apt install ffmpeg -y

python -m server_ggpuf
```

Redirect to `http://localhost:8000/docs` to see the API documentation and test the service.

We are working on making this process more straightforward. Stay tuned for updates!
