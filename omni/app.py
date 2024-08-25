import gradio as gr
import numpy as np

import numpy as np
import torch

from encodec.utils import save_audio

from common import DEVICE, ctx, TEXT, SEMANTIC
from common import Config as cfg

from common import Config as cfg
from tts.infer import AudioSemantic

from tts.hfload import convert_to_hf
from huggingface_hub import snapshot_download

from common import cache_dir

local_dir = f'{cache_dir}/models/omni_774m_tinystories'
tokenizer = AudioSemantic(size='125m')
snapshot_download(f'cmeraki/omni_774m_tinystories', local_dir=local_dir)
omni_model = convert_to_hf(path=f'{local_dir}/omni.pt', device=DEVICE)


def decorate(tokens, type):
    tokens = tokens + cfg.OFFSET[type]
    tokens = np.hstack([cfg.INFER_TOKEN[type],
                        tokens,
                        cfg.STOP_TOKEN[type]])
    return tokens

def converse(text):
    human_text = text

    human_text_tokens = tokenizer.text_tokenizer.encode(human_text)
    human_text_tokens = np.asarray(human_text_tokens)
    human_text_tokens = decorate(human_text_tokens, type=TEXT)

    alltokens = np.hstack([human_text_tokens, [cfg.INFER_TOKEN[SEMANTIC]]])

    omni_model.generation_config.eos_token_id = cfg.STOP_TOKEN[SEMANTIC]

    input_tokens = (torch.tensor(alltokens,
                                 dtype=torch.long,
                                 device=DEVICE)[None, ...])
    with ctx:
        target_tokens = omni_model.generate(input_tokens,
                                       max_length=1024,
                                       temperature=0.7,
                                       top_k=100,
                                       do_sample=True)

        target_tokens = target_tokens.detach().cpu().numpy()[0]
        target_tokens = target_tokens[len(alltokens):]

    end_idx = np.where(target_tokens == cfg.STOP_TOKEN[SEMANTIC])[0][0]
    target_tokens = target_tokens[0:end_idx]
    target_tokens = target_tokens - cfg.OFFSET[SEMANTIC]

    next_audio = tokenizer.semantic_to_audio(target_tokens)
    tmp_audio_file = f'omni.wav'
    save_audio(next_audio[0], tmp_audio_file, sample_rate=24000)
    return 24_000, next_audio[0][0].cpu().numpy()
    

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
    demo.launch()