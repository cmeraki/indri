import torch
from pathlib import Path
from contextlib import nullcontext
from configs.constants import *

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
rng_state = torch.random.get_rng_state()

DEVICE = 'cuda:0'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
device_type = 'cuda' if 'cuda' in DEVICE else 'cpu'
ptdtype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[dtype]

CTX = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)
CACHE_DIR = Path("~/.cache/indri/").expanduser()
Path(CACHE_DIR).mkdir(exist_ok=True, parents=True)

print('Cache directory at: ', CACHE_DIR)

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

    TASK_TOKENS = {
        CONVERT: '[convert]',
        CONTINUE: '[continue]',
    }

    MODALITY_TOKENS = {
        TEXT: '[text]',
        SEMANTIC: '[semantic]',
        ACOUSTIC: '[acoustic]',
    }

    # This stop token is used for all the modalities
    STOP_TOKEN = '[stop]'
    VOCAB_SIZE = (sum(VOCAB_SIZES.values()) // 64 + 1)*64

    print('Gap tokens: ', VOCAB_SIZE - sum(VOCAB_SIZES.values()))
