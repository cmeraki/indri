from configs.training_commons import *

# Training specific configs
STEPS = 100000
EVAL_INTERVAL = 500
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 16

MODEL_TYPE = 'gpt2-medium'
MAX_SOURCE_TOKENS = 768
BLOCK_SIZE = 3072
