import torch
import random
import numpy as np
from pathlib import Path
from contextlib import nullcontext

MIMI = 'mimi'
TEXT = 'text'
AUDIO = 'audio'
ANNOTATIONS = 'annotation'
TOKENS = 'tokens'
CONVERT = 'convert'
CONTINUE = 'continue'

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
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

SPEAKER_FILE = 'allowed_speakers.jsonl'

class Config:
    n_codebooks = 8
    per_codebook_size = 2048

    VOCAB_SIZES = {
        TEXT: 50257,
        MIMI: per_codebook_size * n_codebooks,
    }
    
    OFFSET = {
        TEXT: 0,
        MIMI: VOCAB_SIZES[TEXT],
    }
    
    TASK_TOKENS = {
        CONVERT: '[convert]',
        CONTINUE: '[continue]',
    }

    MODALITY_TOKENS = {
        TEXT: '[text]',
        MIMI: '[mimi]',
    }

    UNKNOWN_SPEAKER_ID = '[spkr_unk]'

    # This stop token is used for all the modalities
    STOP_TOKEN = '[stop]'
    VOCAB_SIZE = 70016

    print('Gap tokens: ', VOCAB_SIZE - sum(VOCAB_SIZES.values()))
