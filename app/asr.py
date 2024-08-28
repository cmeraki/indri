import pdb
import torch
from torch.cuda import empty_cache
import numpy as np
import gradio as gr
from functools import partial
from transformers import Wav2Vec2FeatureExtractor
from audiotoken import AudioToken, Tokenizers

from common import SEMANTIC, TEXT, cache_dir, device
from tts.infer import generate
from omni.hfload import convert_to_hf
from tts.infer import AudioSemantic as VanillaAudioSemantic
from tts.utils import replace_consecutive, convert_audio

ttslib = VanillaAudioSemantic()

semantic_tokenizer = AudioToken(Tokenizers.semantic_s, device='cuda:0')
model_dir = f'{cache_dir}/models/tts_en_xl_125m/'
semantic_text_model = convert_to_hf(path=f'{model_dir}/semantic_text/gpt_last.pt', device=device)

def hubert_processor(audio, processor):
    return processor(
        audio,
        sampling_rate=16_000,
        return_tensors='pt'
    ).input_values[0]


processor = Wav2Vec2FeatureExtractor.from_pretrained('voidful/mhubert-base')
transform_func = partial(hubert_processor, processor=processor)

def transcribe_audio(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y = y.reshape(1, -1)
    y = torch.tensor(y)

    print(y.shape)

    aud = convert_audio(y, sr, target_sr=16_000, target_channels=1)
    aud = transform_func(aud)

    print(aud.shape)

    source_tokens = semantic_tokenizer.encode(aud)
    source_tokens = source_tokens.cpu().numpy()[0][0]
    source_tokens = replace_consecutive(source_tokens)

    print(f"Source tokens shape: {source_tokens.shape}")

    txt_toks =  generate(
        model=semantic_text_model,
        source_tokens=source_tokens,
        source=SEMANTIC,
        target=TEXT,
        max_length=1024,
        max_source_tokens=768,
        temperature=0.8,
        top_k=100
    )

    text = ttslib.text_tokenizer.decode(txt_toks)

    empty_cache()

    return text


demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy"),
    outputs="text",
    live=True,
    title="Automatic Speech Recognition",
    description="Speak into your microphone to see the transcription."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7880)

# if __name__ == "__main__":
#     transcribe_audio((16_000, np.random.rand(16000)))