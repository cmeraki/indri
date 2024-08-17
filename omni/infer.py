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
from omni.instructlib import HUMAN, ASSISTANT
from tts.infer import AudioSemantic
from omni.instructlib import to_text_tokens
import simpleaudio as sa


def playwav(wav):
    wave_obj = sa.WaveObject.from_wave_file(wav)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def load_model(path):
    model = GPT.from_pretrained('cmeraki/gpt2-124M-400B')
    model.expand_vocab(new_vocab_size=cfg.VOCAB_SIZE)
    model.to(DEVICE)

    state_dict = torch.load(path)['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    
    model.eval()
    return model

def extract_new_tokens(y, target):
    start_idx = np.where(y == cfg.INFER_TOKEN[target])[0]
    end_idx = np.where(y == cfg.STOP_TOKEN[target])[0]
    if end_idx.any():
        y = y[start_idx[0] + 1: end_idx[0]]
    else:
        y = y[start_idx[0] + 1:]

    return y

def generate(model, tokens, stop_token):
    input_tokens = (torch.tensor(tokens,
                                dtype=torch.long,
                                device=DEVICE)[None, ...])
    
    
    with torch.no_grad():
        with ctx:
            target_tokens = model.generate(input_tokens, 
                                1024, 
                                temperature=0.8,
                                top_k=100, 
                                stop_token=stop_token)
            
            target_tokens = target_tokens.detach().cpu().numpy()[0]
    
    return target_tokens

def generate_next(prompt, modality, omni_model):
    alltokens = np.hstack(prompt  + [cfg.INFER_TOKEN[modality]])
    # print("input", alltokens)

    result = generate(model=omni_model, 
                      tokens=alltokens, 
                      stop_token=cfg.STOP_TOKEN[modality])
    
    result = result[len(alltokens):] - cfg.OFFSET[modality]
    return result

def converse():
    omni_model = load_model(path='/home/apurva/.cache/indri/data/models/omni/omni/gpt_500.pt')

    tokenizer = AudioSemantic(size='125m')
    
    human_token = tokenizer.text_tokenizer.encode(HUMAN)
    assistant_token = tokenizer.text_tokenizer.encode(ASSISTANT)

    print('human_token', human_token, 'assistant_token', assistant_token)
    for i in range(100):
        if i > 0:
            human_text = input('human:')
        else:
            human_text = 'warmup'
        
        human_text_tokens = to_text_tokens(human_text, tokenizer)
        human_prompt = [human_token, human_text_tokens, assistant_token]
        next_text_tokens = generate_next(human_prompt, TEXT, omni_model=omni_model)
        next_semantic_tokens = generate_next(human_prompt, SEMANTIC, omni_model=omni_model)
        
        next_text = tokenizer.text_tokenizer.decode(next_text_tokens)
        print(next_text)

        # uncomment to speak text via tts
        # next_semantic_tokens = tokenizer.text_to_semantic(next_text)
        

        next_audio = tokenizer.semantic_to_audio(next_semantic_tokens)
        tmp_audio_file = '/tmp/test.wav'
        save_audio(next_audio[0], tmp_audio_file, sample_rate=24000)
        playwav(tmp_audio_file)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--size', default='125m', required=False)
    parser.add_argument('--text', default='this is a test <comma> one you should not fail <period>, if you fail there will be consequences, those consequences are not imaginable', required=False)
    parser.add_argument('--output', default='test.wav', required=False)
    
    args = parser.parse_args()
    converse()