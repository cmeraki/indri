import os
import json

os.environ["SUNO_OFFLOAD_CPU"] = "False"

import numpy as np
import bark
import torch
from pathlib import Path

from encodec.utils import save_audio

from tts.gpt2_model import get_model

from datalib.tokenlib import get_tokenizer
from tts.train_tts import cfg, DataLoader
from common import SEMANTIC, TEXT, ACOUSTIC, device, ctx

class GPTModel:
    def __init__(self, path, source, target, device='cuda:0'):
        self.device = device

        self.path = Path(path)/'gpt_last.pt'
        self.vocab_size = cfg.VOCAB_SIZE
        self.model = self.load(self.path)

        self.source = source
        self.target = target

    def load(self, path):
        model = get_model(vocab_size=self.vocab_size, device=self.device, compile=False, path=path)
        model.eval()
        return model

    def generate(self, tokens, max_new_tokens=1024, temperature=0.3, top_k=100, stop_token=None):
        with torch.no_grad():
            with ctx:
                y = self.model.generate(tokens, 
                                        max_new_tokens, 
                                        temperature=temperature,
                                        top_k=top_k, 
                                        stop_token=stop_token)
                
                y = y.detach().cpu().numpy()[0]
        return y


def prepare_input(source_tokens, target_prompt, source, target, prompt_length):
    source_tokens = DataLoader.prepare_source(source_tokens,
                                            source=source,
                                            max_source_tokens=cfg.max_source_tokens)
    
    prompt_arr = DataLoader.prepare_prompt(prompt=None,
                                            target=target,
                                            prompt_length=prompt_length)
    
    source_tokens = np.hstack([source_tokens, prompt_arr])
    source_tokens = (torch.tensor(source_tokens,
                                dtype=torch.long,
                                device=device)[None, ...])
    
    return source_tokens


def run_tts():
    semantic_prompt = np.load('/home/apurva/.cache/huggingface/hub/datasets--cmeraki--gsxl_tokens/snapshots/15630e7e6d09e2db7c12a8d449ec9c0d8877cd62/semantic/AUD0000000007_S0000008.npy')
    acoustic_prompt = np.load('/home/apurva/.cache/huggingface/hub/datasets--cmeraki--gsxl_tokens/snapshots/15630e7e6d09e2db7c12a8d449ec9c0d8877cd62/acoustic/AUD0000000007_S0000008.npy')
    
    text_semantic_model = GPTModel(path='data/models/out_400b_ft_xl/text_semantic/',
                                   source=TEXT,
                                   target=SEMANTIC,
                                   device=device)

    semantic_acoustic_model = GPTModel(path='data/models/out_400b_ft_xl/semantic_acoustic/',
                                       source=SEMANTIC,
                                       target=ACOUSTIC,
                                       device=device)

    text = "MERCUTIO <COMMA> KINSMAN TO THE PRINCE AND FRIEND TO ROMEO <PERIOD>".lower()
    text_tokenizer = get_tokenizer(TEXT, device='cpu')
    text_tokens = np.asarray(text_tokenizer.encode(text))
    print(text_tokens)

    for i in range(100):
        input_tokens = prepare_input(text_tokens,
                                     semantic_prompt,
                                     source=TEXT, 
                                     target=SEMANTIC,
                                     prompt_length=cfg.PROMPT_LENGTH[SEMANTIC])
        
        print("input", input_tokens)
        
        semantic_tokens = text_semantic_model.generate(input_tokens, stop_token=cfg.PAD_TOKEN[SEMANTIC])
        input_length = cfg.max_source_tokens + cfg.PROMPT_LENGTH[SEMANTIC]
        semantic_tokens = semantic_tokens[input_length:] - cfg.OFFSET[SEMANTIC]
        
        print("p", list(semantic_tokens))
        
        input_tokens = prepare_input(semantic_tokens,
                                     acoustic_prompt,
                                     source=SEMANTIC,
                                     target=ACOUSTIC,
                                     prompt_length=cfg.PROMPT_LENGTH[ACOUSTIC])
        
        print("semantic", input_tokens.shape)
        acoustic_tokens = semantic_acoustic_model.generate(input_tokens, stop_token=cfg.PAD_TOKEN[ACOUSTIC])
        input_length = cfg.max_source_tokens + cfg.PROMPT_LENGTH[ACOUSTIC]
        acoustic_tokens = acoustic_tokens[input_length:] - cfg.OFFSET[ACOUSTIC]

        print("acoustic", acoustic_tokens)

        acoustic_tokenizer = get_tokenizer(ACOUSTIC, device='cpu')
        wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        save_audio(wav[0], f'tts_{i}.wav', sample_rate=24000)


if __name__ == "__main__":
    run_tts()