import torch
from pathlib import Path
import os
from contextlib import nullcontext


SEMANTIC = 'semantic'
ACOUSTIC = 'acoustic'
TEXT = 'text'
AUDIO = 'audio'
IMAGE = 'image'
ANNOTATIONS = 'annotation'
TOKENS = 'tokens'

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
rng_state = torch.random.get_rng_state()

print(f'RNG state: {rng_state}')

device = 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

DEVICE = 'cuda:0'

cache_dir = os.path.expanduser("~/.cache/indri/")
Path(cache_dir).mkdir(exist_ok=True, parents=True)
print('cache at', cache_dir)


class Config:
    coarse_codebooks = 2
    per_codebook_size = 1024

    VOCAB_SIZES = {
        TEXT: 50257,
        SEMANTIC: 1000,
        ACOUSTIC: 2048,
    }

    OFFSET = {
        TEXT: 0,
        SEMANTIC: VOCAB_SIZES[TEXT],
        ACOUSTIC: VOCAB_SIZES[TEXT] + VOCAB_SIZES[SEMANTIC],
    }
    
    max_token_value = 0
    for i in OFFSET:
        max_token_value = max(OFFSET[i] + VOCAB_SIZES[i], max_token_value)
        

    PAD_TOKEN = {
        TEXT: 50256,
        SEMANTIC: max_token_value + 2,
        ACOUSTIC: max_token_value + 3,
    }

    INFER_TOKEN = {
        TEXT: max_token_value + 4,
        SEMANTIC: max_token_value + 5,
        ACOUSTIC: max_token_value + 6
    }

    STOP_TOKEN = {
        TEXT: max_token_value + 7,
        SEMANTIC: max_token_value + 8,
        ACOUSTIC: max_token_value + 9,
    }

    VOCAB_SIZE = (max(STOP_TOKEN.values()) // 64 + 1)*64
    
    print('GAP tokens =', VOCAB_SIZE - max(STOP_TOKEN.values()))

    max_source_tokens = 768
    
    PROMPT_LENGTH = {
        SEMANTIC : 0,
        ACOUSTIC : 0
    }

    OMNI_STOP_TOKEN = VOCAB_SIZE - 1

    

    
