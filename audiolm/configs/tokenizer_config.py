OFFSET = "offset"
PAD_TOKEN = "pad_token"
VOCAB_SIZE = "vocab_sizes"
EXPANDED_VOCAB_SIZE = "expanded_vocab_size"

SEMANTIC = 'semantic'
ACOUSTIC = 'acoustic'
TEXT = 'text'
IMAGE = 'image'

coarse_codebooks = 2
per_codebook_size = 1024

def calculate_vocab_config_gpt2(source, target):
    _VOCAB_SIZES = {
        TEXT: 50257,
        SEMANTIC: 1000,
        ACOUSTIC: 2048,
        IMAGE: 8192
    }

    config = {
        VOCAB_SIZE: _VOCAB_SIZES,

        OFFSET: {
            source: 0,
            target: _VOCAB_SIZES[source],
        },
        PAD_TOKEN: {
            source: _VOCAB_SIZES[source] + _VOCAB_SIZES[target] + 1,
            target: _VOCAB_SIZES[source] + _VOCAB_SIZES[target] + 2,
        }
    }


    config[PAD_TOKEN][TEXT] = _VOCAB_SIZES[TEXT] - 1
    config[EXPANDED_VOCAB_SIZE] = max(max(config[VOCAB_SIZE][s] + config[OFFSET][s], config[PAD_TOKEN][s]) for s in [source, target]) + 1
    return config
