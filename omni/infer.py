import numpy as np
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from encodec.utils import save_audio

from tts.gpt2_model import get_model, GPT
from tts.train import DataLoader
from common import DEVICE, ctx, TEXT, SEMANTIC
from common import Config as cfg
from datalib.tokenlib import get_tokenizer
from common import cache_dir

from common import Config as cfg
from tts.utils import read_audio_file
from tts.infer import AudioSemantic
import simpleaudio as sa


def playwav(wav):
    wave_obj = sa.WaveObject.from_wave_file(wav)
    play_obj = wave_obj.play()
    play_obj.wait_done()


def extract_new_tokens(y, target):
    start_idx = np.where(y == cfg.INFER_TOKEN[target])[0]
    end_idx = np.where(y == cfg.STOP_TOKEN[target])[0]
    if end_idx.any():
        y = y[start_idx[0] + 1: end_idx[0]]
    else:
        y = y[start_idx[0] + 1:]

    return y


def generate_next(prompt, modality, model):
    alltokens = np.hstack([prompt, [cfg.INFER_TOKEN[modality]]])

    model.generation_config.eos_token_id = cfg.STOP_TOKEN[modality]
    # model.generation_config.pad_token_id = cfg.STOP_TOKEN[modality]

    input_tokens = (torch.tensor(alltokens,
                                 dtype=torch.long,
                                 device=DEVICE)[None, ...])

    print(input_tokens)
    with ctx:
        target_tokens = model.generate(input_tokens,
                                       max_length=1024,
                                       temperature=0.7,
                                       top_k=100,
                                       do_sample=True)

        target_tokens = target_tokens.detach().cpu().numpy()[0]
        target_tokens = target_tokens[len(alltokens):]

    end_idx = np.where(target_tokens == cfg.STOP_TOKEN[modality])[0][0]
    target_tokens = target_tokens[0:end_idx]
    print(f'generated_{modality}', target_tokens)
    # target_tokens = extract_new_tokens(target_tokens, target=modality)

    target_tokens = target_tokens - cfg.OFFSET[modality]

    return target_tokens


def decorate(tokens, type):
    tokens = tokens + cfg.OFFSET[type]
    tokens = np.hstack([cfg.INFER_TOKEN[type],
                        tokens,
                        cfg.STOP_TOKEN[type]])
    return tokens


def converse():
    from tts.hfload import convert_to_hf
    omni_model = convert_to_hf(path=f'/home/meraki/Downloads/omni.pt', device=DEVICE)
    print(omni_model)

    tokenizer = AudioSemantic(size='125m')

    human_text = 'once upon a time there was a girl named mary'

    human_text_tokens = tokenizer.text_tokenizer.encode(human_text)
    human_text_tokens = np.asarray(human_text_tokens)
    human_text_tokens = decorate(human_text_tokens, type=TEXT)
    t = tokenizer.text_tokenizer.decode(human_text_tokens)
    # print("DC", t)

    for i in range(10):
        next_text_tokens = generate_next(human_text_tokens, TEXT, model=omni_model)
        next_semantic_tokens = generate_next(human_text_tokens, SEMANTIC, model=omni_model)

        next_text = tokenizer.text_tokenizer.decode(next_text_tokens)

        print("text", next_text)
        print("semantic", list(next_semantic_tokens[:256]), next_semantic_tokens.shape)

        # uncomment to speak text via tts
        # next_semantic_tokens = tokenizer.text_to_semantic(next_text)

        next_audio = tokenizer.semantic_to_audio(next_semantic_tokens)
        tmp_audio_file = f'omni_{i}.wav'
        save_audio(next_audio[0], tmp_audio_file, sample_rate=24000)
        playwav(tmp_audio_file)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--size', default='125m', required=False)
    parser.add_argument('--text',
                        default='this is a test <comma> one you should not fail <period>, if you fail there will be consequences, those consequences are not imaginable',
                        required=False)
    parser.add_argument('--output', default='test.wav', required=False)

    args = parser.parse_args()
    converse()