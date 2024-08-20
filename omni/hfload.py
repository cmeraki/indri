import torch
from transformers import GPT2LMHeadModel, GPT2Config

from tts.gpt2_model import get_model

# inverse of : https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L207
# converts a saved nanogpt model back to hf format so it can be loaded in vllm

def load_model(path):
    print(path)
    model = get_model(
        vocab_size=53376,
        device='cpu',
        compile=False,
        path=path
    )

    model.eval()
    return model

def load_as_hf_model(path):
    custom_gpt = load_model(path)
    custom_gpt = custom_gpt.state_dict()

    config = GPT2Config(
        vocab_size=53376,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        use_bias=False,
        dropout=0,
        attn_pdrop=0,
        embd_pdrop=0,
        resid_pdrop=0,
        summary_first_dropout=0,
        activation_function='gelu'
    )

    clean_custom_gpt = {}
    unwanted_prefix = '_orig_mod.'
    for k, v in custom_gpt.items():
        if k.startswith(unwanted_prefix):
            clean_custom_gpt[k[len(unwanted_prefix):]] = custom_gpt[k]

    transposed = [
        'attn.c_attn.weight',
        'attn.c_proj.weight',
        'mlp.c_fc.weight',
        'mlp.c_proj.weight'
    ]

    for k, v in clean_custom_gpt.items():
        if any(k.endswith(w) for w in transposed):
            clean_custom_gpt[k] = clean_custom_gpt[k].t()
