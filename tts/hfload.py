import torch
from transformers import GPT2LMHeadModel, GPT2Config

def convert_to_hf(path, device: str = 'cpu'):
    custom_gpt = torch.load(path, map_location=device)['model']
    clean_custom_gpt = {}
    unwanted_prefix = '_orig_mod.'
    for k, _ in custom_gpt.items():
        if k.startswith(unwanted_prefix):
            clean_custom_gpt[k[len(unwanted_prefix):]] = custom_gpt[k]

    transposed = [
        'attn.c_attn.weight',
        'attn.c_proj.weight',
        'mlp.c_fc.weight',
        'mlp.c_proj.weight'
    ]

    for k, _ in clean_custom_gpt.items():
        if any(k.endswith(w) for w in transposed):
            clean_custom_gpt[k] = clean_custom_gpt[k].t()

    custom_gpt_config = torch.load(path, map_location=device)['config']

    model_args = dict(
        use_bias=False,
        dropout=0,
        attn_pdrop=0,
        embd_pdrop=0,
        resid_pdrop=0,
        summary_first_dropout=0,
        activation_function='gelu',
    )
    
    config = GPT2Config(**model_args)
    
    config.n_layer = custom_gpt_config.n_layer
    config.n_head = custom_gpt_config.n_head
    config.vocab_size = custom_gpt_config.vocab_size
    config.n_embd = custom_gpt_config.n_embd
    config.n_positions = custom_gpt_config.block_size

    model = GPT2LMHeadModel(config)
    model.to(device)
    model.load_state_dict(clean_custom_gpt, strict=False)

    return model

