# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from datalib import VOCAB_SIZES, PAD_TOKEN, OFFSET
from tokenlib import EncodecTokenizer, SEMANTIC, ACOUSTIC, TEXT, IMAGE
from gpt2_model import get_model

from contextlib import nullcontext
import os
from tqdm import tqdm
from train_image_gen import get_vocab_size
from tokenlib import get_tokenizer

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


class GPTModel:
    def __init__(self, source, target, device='cuda:0'):
        self.source = source
        self.target = target
        self.device = device

        self.path = f'out/{source}_{target}/gpt_20000.pt'
        self.model = self.load(self.path)
        self.vocab_size = get_vocab_size(source=source, target=target)

    def load(self, path):
        saved_model = torch.load(path)['model']

        model = get_model(n_layer=4,
                        n_head=4,
                        n_embd=256,
                        vocab_size=get_vocab_size(self.source, self.target),
                        block_size=1280,
                        compile=True,
                        device=self.device)
        
        model.load_state_dict(saved_model)
        model.eval()
        return model

    def generate(self, tokens, max_new_tokens=1088, temperature=0.2, top_k=30):
        with torch.no_grad():
            with ctx:
                y = self.model.generate(tokens, max_new_tokens, temperature=temperature, top_k=top_k)
                y = y.detach().cpu().numpy()[0]
                start_idx = np.where(y == PAD_TOKEN[self.source])[0][0]
                # end_idx = np.where(y == PAD_TOKEN[self.target])[0][0]
                
                y = y[start_idx + 1: start_idx + 1025]
        
        return y


def run_image():
    text_image_model = GPTModel(source=TEXT, target=IMAGE, device=device)
    
    text = "she climbed a tree and fell"
    text_tokenizer = get_tokenizer(TEXT, device='cpu')
    text_tokens = np.asarray(text_tokenizer.encode(text)) + OFFSET[TEXT]
    
    text_tokens = np.append(text_tokens, PAD_TOKEN[TEXT])
    text_tokens = (torch.tensor(text_tokens, dtype=torch.long, device=device)[None, ...])
    
    image_tokenizer = get_tokenizer(IMAGE, device=device)
    
    for i in range(100):
        image_tokens = text_image_model.generate(text_tokens)
        
        image_tokens = torch.tensor(image_tokens[:1024] - OFFSET[IMAGE]).to(device)
        print(image_tokens.size(), image_tokens)
        img = image_tokenizer.decode(image_tokens)
        img.save(f'test_{i}.png')

if __name__ == "__main__":
    run_image()