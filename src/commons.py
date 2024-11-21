import torch
import random
import numpy as np

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
