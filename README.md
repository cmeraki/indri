[![Live Demo](https://img.shields.io/badge/🌎%20Live%20Demo-indrivoice.ai-brightgreen)](https://indrivoice.ai/)
[![Twitter](https://img.shields.io/badge/𝕏%20Twitter-@11mlabs_in-black)](https://x.com/11mlabs_in)
[![Hugging Face Collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/11mlabs/indri-673dd4210b4369037c736bfe)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Server-yellow)](https://huggingface.co/spaces/11mlabs/IndriVoice)
[![Release Blog](https://img.shields.io/badge/📝%20Release%20Blog-2024--11--21-lightgrey)](https://www.indrivoice.ai/blog/2024-11-21-building-indri-tts)

# Indri

Indri is a series of multilingual audio models that can do TTS, ASR, and audio continuation. It currently supports these languages:

1. English
2. Hindi

This repo hosts the inference code for inference of Indri models.

## Samples

| Text | Sample |
| --- | --- |
|मित्रों, हम आज एक नया छोटा और शक्तिशाली मॉडल रिलीज कर रहे हैं।| <audio controls src="https://huggingface.co/11mlabs/indri-0.1-124m-tts/resolve/main/data/cebed668-62cb-4188-a2e1-3af8e017d3ba.wav" title="Title"></audio> |
|भाइयों और बहनों, ये हमारा सौभाग्य है कि हम सब मिलकर इस महान देश को नई ऊंचाइयों पर ले जाने का सपना देख रहे हैं।| <audio controls src="https://huggingface.co/11mlabs/indri-0.1-124m-tts/resolve/main/data/6e0a4879-0379-4166-a52c-03220a3f2922.wav" title="Title"></audio> |
|Hello दोस्तों, future of speech technology mein अपका स्वागत है | <audio controls src="https://huggingface.co/11mlabs/indri-0.1-124m-tts/resolve/main/data/5848b722-efe3-4e1f-a15e-5e7d431cd475.wav" title="Title"></audio> |
|In this model zoo, a new model called Indri has appeared.| <audio controls src="https://huggingface.co/11mlabs/indri-0.1-124m-tts/resolve/main/data/7ac0df93-edbd-47b2-b850-fb88e329998c.wav" title="Title"></audio> |

## Key features

1. Extremely small, based on GPT-2 small architecture. The methodology can be extended to any autoregressive transformer-based architecture.
2. Ultra-fast. Using our [self hosted service option](#self-hosted-service), on RTX6000Ada NVIDIA GPU the model can achieve speeds up to 400 toks/s (4s of audio generation per s) and under 20ms time to first token for the 124m model.
3. On RTX6000Ada, it can support a batch size of ~1000 sequences with full context length of 1024 tokens
4. Supports voice cloning with small prompts (<5s).
5. Code mixing text input in 2 languages - English and Hindi.

## Details

1. Model Type: GPT-2 based language model
2. Size: 124M parameters
3. Language Support: English, Hindi
4. License: This model is not for commercial usage. This is only a research showcase.

Here's a brief of how the model works:

1. Converts input text into tokens
2. Runs autoregressive decoding on GPT-2 based transformer model and generates audio tokens
3. Decodes audio tokens (using [Kyutai/mimi](https://huggingface.co/kyutai/mimi)) to audio

Please read our blog [here](https://www.indrivoice.ai/blog/2024-11-21-building-indri-tts) for more technical details on how it was built.

## How to Get Started with the Model

### 🤗 pipelines

Use the code below to get started with the model. Pipelines are the best way to get started with the model.

```python
import torch
import torchaudio
from transformers import pipeline

model_id = '11mlabs/indri-0.1-124m-tts'
task = 'indri-tts'

pipe = pipeline(
    task,
    model=model_id,
    device=torch.device('cuda:0'), # Update this based on your hardware,
    trust_remote_code=True
)

output = pipe(['Hi, my name is Indri and I like to talk.'], speaker = '[spkr_63]')

torchaudio.save('output.wav', output[0]['audio'][0], sample_rate=24000)
```

### Available speakers

|Speaker ID|Speaker name|
|---|---|
|`[spkr_63]`|🇬🇧 👨 book reader|
|`[spkr_67]`|🇺🇸 👨 influencer|
|`[spkr_68]`|🇮🇳 👨 book reader|
|`[spkr_69]`|🇮🇳 👨 book reader|
|`[spkr_70]`|🇮🇳 👨 motivational speaker|
|`[spkr_62]`|🇮🇳 👨 book reader heavy|
|`[spkr_53]`|🇮🇳 👩 recipe reciter|
|`[spkr_60]`|🇮🇳 👩 book reader|
|`[spkr_74]`|🇺🇸 👨 book reader|
|`[spkr_75]`|🇮🇳 👨 entrepreneur|
|`[spkr_76]`|🇬🇧 👨 nature lover|
|`[spkr_77]`|🇮🇳 👨 influencer|
|`[spkr_66]`|🇮🇳 👨 politician|

### Self hosted service

```bash
git clone https://github.com/cmeraki/indri.git
cd indri
pip install -r requirements.txt

# Install ffmpeg (for Mac/Windows, refer here: https://www.ffmpeg.org/download.html)
sudo apt update -y
sudo apt upgrade -y
sudo apt install ffmpeg -y

python -m server --model_path 11mlabs/indri-0.1-124m-tts --device cuda:0 --port 8000
```

Defaults:

- `device`: `cuda:0`
- `port`: `8000`

Choices:

- `model_path`: [HuggingFace collection](https://huggingface.co/collections/11mlabs/indri-673dd4210b4369037c736bfe)

Redirect to `http://localhost:8000/docs` to see the API documentation and test the service.

To run the GGUF quantized models, follow the instructions [here](src/README.GGUF.md).

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{indri-multimodal-alm,
  author       = {11mlabs},
  title        = {Indri: Multimodal audio language model},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished = {\url{https://github.com/cmeraki/indri}},
  email        = {compute@merakilabs.com}
}
```

## BibTex

1. [nanoGPT](https://github.com/karpathy/nanoGPT)

2. [Kyutai/mimi](https://huggingface.co/kyutai/mimi)

```bibtex
@techreport{kyutai2024moshi,
      title={Moshi: a speech-text foundation model for real-time dialogue},
      author={Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and
      Am\'elie Royer and Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
      year={2024},
      eprint={2410.00037},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.00037},
}
```

3. [Whisper](https://github.com/openai/whisper)

```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

4. [silero-vad](https://github.com/snakers4/silero-vad)

```bibtex
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```
