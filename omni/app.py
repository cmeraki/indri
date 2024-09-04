import torch
import gradio as gr
import numpy as np
from huggingface_hub import snapshot_download
from encodec.utils import save_audio

from omni.hfload import convert_to_hf
from datalib.tokenlib import get_tokenizer
from common import Config as cfg, ctx, cache_dir, DEVICE, TEXT, SEMANTIC, ACOUSTIC

local_dir = f'{cache_dir}/models/omni_774m_tinystories'
snapshot_download(f'cmeraki/omni_774m_tinystories', local_dir=local_dir)
omni_model = convert_to_hf(path=f'{local_dir}/omni.pt', device=DEVICE)

text_tokenizer = get_tokenizer(TEXT, device='cpu')
acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)

local_dir = f'{cache_dir}/models/tts_xl_30k_long_125m_en/semantic_acoustic/'
snapshot_download(f'cmeraki/tts_xl_30k_long_125m_en/semantic_acoustic/', local_dir=local_dir)
semantic_acoustic_model = convert_to_hf(path=f'{local_dir}/gpt_last.pt', device=DEVICE)

def decorate(tokens, type):
    tokens = tokens + cfg.OFFSET[type]
    tokens = np.hstack([cfg.INFER_TOKEN[type],
                        tokens,
                        cfg.STOP_TOKEN[type]])
    return tokens

def extract_new_tokens(y, target):
    start_idx = np.where(y == cfg.INFER_TOKEN[target])[0]
    end_idx = np.where(y == cfg.STOP_TOKEN[target])[0]
    if end_idx.any():
        y = y[start_idx[0] + 1: end_idx[0]]
    else:
        y = y[start_idx[0] + 1:]

    return y

def converse(text):
    human_text = text

    human_text_tokens = text_tokenizer.encode(human_text)
    human_text_tokens = np.asarray(human_text_tokens)
    human_text_tokens = decorate(human_text_tokens, type=TEXT)

    alltokens = np.hstack([human_text_tokens, [cfg.INFER_TOKEN[SEMANTIC]]])

    omni_model.generation_config.eos_token_id = cfg.STOP_TOKEN[SEMANTIC]

    input_tokens = (torch.tensor(alltokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Text -> Semantic/Text
    with ctx:
        semantic_tokens = omni_model.generate(
            input_tokens,
            max_length=1024,
            temperature=0.7,
            top_k=100,
            do_sample=True
        )

        semantic_tokens = semantic_tokens.detach().cpu().numpy()[0]
        semantic_tokens = semantic_tokens[len(alltokens):]

    max_source_semantic_tokens = cfg.MAX_SOURCE_TOKENS[SEMANTIC]
    end_idx = np.where(semantic_tokens == cfg.STOP_TOKEN[SEMANTIC])[0][0]
    semantic_tokens = semantic_tokens[0:end_idx]
    semantic_tokens = semantic_tokens[:max_source_semantic_tokens]

    semantic_tokens = np.hstack([semantic_tokens, cfg.INFER_TOKEN[ACOUSTIC]])
    semantic_tokens = (torch.tensor(semantic_tokens, dtype=torch.long,device=DEVICE)[None, ...])

    # Semantic -> Acoustic
    with ctx:
        acoustic_tokens = semantic_acoustic_model.generate(
            semantic_tokens,
            max_length=cfg.BLOCK_SIZE[SEMANTIC],
            temperature=0.95,
            top_k=100,
            do_sample=True
        )
        acoustic_tokens = acoustic_tokens.detach().cpu().numpy()[0]

    acoustic_tokens = extract_new_tokens(acoustic_tokens, target=ACOUSTIC)
    acoustic_tokens = acoustic_tokens - cfg.OFFSET[ACOUSTIC]
    if len(acoustic_tokens) % 2 == 1:
        acoustic_tokens = acoustic_tokens[:-1]

    # Acoustic -> Audio
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    wav = wav[0].cpu()
    tmp_audio_file = f'omni.wav'
    save_audio(wav, tmp_audio_file, sample_rate=24000)

    return 24_000, wav[0].numpy()


demo = gr.Interface(
    fn=converse,
    inputs=[
        gr.Textbox(label="Text to continue"),
    ],
    outputs=gr.Audio(label="Generated Audio", autoplay=True),
    title="Text-Audio Multimodal",
    description="Enter text to continue audio"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
