import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"

import numpy as np
import bark
import torch
from encodec.utils import save_audio
from datalib import VOCAB_SIZES, PAD_TOKEN, OFFSET
from tokenlib import EncodecTokenizer, SEMANTIC, ACOUSTIC, TEXT
from gpt2_model import get_model

from contextlib import nullcontext
import os
from tqdm import tqdm
from train_tts import get_vocab_size
from tokenlib import get_tokenizer

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'

device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


class GPTModel:
    def __init__(self, source, target, device='cuda:0'):
        self.source = source
        self.target = target
        self.device = device

        self.path = f'out/{source}_{target}/gpt_last.pt'
        self.model = self.load(self.path)
        self.vocab_size = get_vocab_size(source=source, target=target)


    def load(self, path):
        saved_model = torch.load(path)['model']

        model = get_model(n_layer=8,
                        n_head=8,
                        n_embd=512,
                        vocab_size=get_vocab_size(self.source, self.target),
                        block_size=1024,
                        compile=True,
                        device=self.device)
        
        model.load_state_dict(saved_model)
        model.eval()
        return model

    def generate(self, tokens, max_new_tokens=1024, temperature=0.8, top_k=100):
        with torch.no_grad():
            with ctx:
                y = self.model.generate(tokens, max_new_tokens, temperature=temperature, top_k=top_k)
                y = y.detach().cpu().numpy()[0]
                start_idx = np.where(y == PAD_TOKEN[self.source])[0][0]
                end_idx = np.where(y == PAD_TOKEN[self.target])[0][0]
                y = y[start_idx + 1: end_idx]
        
        return y

def run_tts():
    text_semantic_model = GPTModel(source=TEXT, target=SEMANTIC, device=device)
    semantic_acoustic_model = GPTModel(source=SEMANTIC, target=ACOUSTIC, device=device)
    
    text = "this was the greatest thing to happen since the big bang"
    text_tokenizer = get_tokenizer(TEXT, device='cpu')
    text_tokens = np.asarray(text_tokenizer.encode(text)) + OFFSET[TEXT]
    
    
    text_tokens = np.append(text_tokens, PAD_TOKEN[TEXT])
    text_tokens = (torch.tensor(text_tokens, dtype=torch.long, device=device)[None, ...])
    semantic_tokens = text_semantic_model.generate(text_tokens)
    
    semantic_tokens = np.append(semantic_tokens, PAD_TOKEN[SEMANTIC])
    semantic_tokens = (torch.tensor(semantic_tokens, dtype=torch.long, device=device)[None, ...])
    acoustic_tokens = semantic_acoustic_model.generate(semantic_tokens)
    
    acoustic_tokenizer = get_tokenizer(ACOUSTIC, device='cpu')
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    save_audio(wav[0], f'tts.wav', sample_rate=24000)

if __name__ == "__main__":
    run_tts()