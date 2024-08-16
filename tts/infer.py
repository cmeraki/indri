import numpy as np
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from encodec.utils import save_audio

from tts.gpt2_model import get_model
from tts.train import DataLoader
from common import SEMANTIC, TEXT, ACOUSTIC, device, ctx
from common import Config as cfg
from datalib.tokenlib import get_tokenizer
from common import cache_dir

from common import Config as cfg

def load_model(path):
    model = get_model(vocab_size=cfg.VOCAB_SIZE, 
                      device=device, 
                      compile=True, 
                      path=path)

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

def generate(model, source, target, source_tokens):
    source_tokens = DataLoader.prepare_source(source_tokens,
                                            source=source,
                                            max_source_tokens=cfg.max_source_tokens)
    
    prompt_arr = DataLoader.prepare_prompt(prompt=None,
                                            target=target,
                                            prompt_length=0)
    
    source_tokens = np.hstack([source_tokens, prompt_arr, cfg.INFER_TOKEN[target]])
    input_tokens = (torch.tensor(source_tokens,
                                dtype=torch.long,
                                device=device)[None, ...])
    
    
    with torch.no_grad():
        with ctx:
            target_tokens = model.generate(input_tokens, 
                                1024, 
                                temperature=0.8,
                                top_k=100, 
                                stop_token=cfg.STOP_TOKEN[target])
            
            target_tokens = target_tokens.detach().cpu().numpy()[0]

    target_tokens = extract_new_tokens(target_tokens, target=target)
        
    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens

class AudioSemantic:
    def __init__(self, size='125m'):
        size = '125m'
        model_dir = f'{cache_dir}/models/tts_en_xl_{size}/'
        snapshot_download(f'cmeraki/tts_en_xl_{size}', local_dir=model_dir)

        self.text_semantic_model = load_model(path=f'{model_dir}/text_semantic/gpt_last.pt')
        self.semantic_acoustic_model = load_model(path=f'{model_dir}/semantic_acoustic/gpt_last.pt')
        self.text_tokenizer = get_tokenizer(TEXT, device='cpu')
        self.acoustic_tokenizer = get_tokenizer(ACOUSTIC, device='cpu')
        
    def text_to_semantic(self, text):
        text_tokens = np.asarray(self.text_tokenizer.encode(text))
        semantic_tokens = generate(model=self.text_semantic_model,
                                   source_tokens=text_tokens, 
                                   source=TEXT, 
                                   target=SEMANTIC)
        return semantic_tokens
        
    def semantic_to_audio(self, tokens):
        acoustic_tokens = generate(model=self.semantic_acoustic_model, 
                                source_tokens=tokens, 
                                source=SEMANTIC, 
                                target=ACOUSTIC)

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--size', default='125m', required=False)
    parser.add_argument('--text', default='this is a test <comma> one you should not fail <period>', required=False)
    parser.add_argument('--output', default='test.wav', required=False)
    
    args = parser.parse_args()
    semlib = AudioSemantic(size=args.size)
    semantic_tokens = semlib.text_to_semantic(args.text)
    wav = semlib.semantic_to_audio(semantic_tokens)
    print("=============")
    print("Writing output to", args.output)
    save_audio(wav=wav[0], path=args.output, sample_rate=24000)
    print("=============")
