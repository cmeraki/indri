import numpy as np
import torch

from encodec.utils import save_audio

from common import DEVICE, ctx, TEXT, SEMANTIC
from common import Config as cfg

from common import Config as cfg
from tts.infer import AudioSemantic

from tts.hfload import convert_to_hf
    


tokenizer = AudioSemantic(size='125m')
omni_model = convert_to_hf(path=f'/home/apurva/Downloads/omni.pt', device=DEVICE)


def decorate(tokens, type):
    tokens = tokens + cfg.OFFSET[type]
    tokens = np.hstack([cfg.INFER_TOKEN[type],
                        tokens,
                        cfg.STOP_TOKEN[type]])
    return tokens

def converse():
    human_text = 'once upon a time there was a girl named mary'

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
    # playwav(tmp_audio_file)


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