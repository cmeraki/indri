import torch
from transformers import GPT2LMHeadModel, GPT2Config

from common import device
from common import Config as cfg

# inverse of : https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L207
# converts a saved nanogpt model back to hf format so it can be loaded in vllm
# hf sampling is different so results are also different

def load_as_hf_model(path):
    config = GPT2Config(vocab_size = cfg.VOCAB_SIZE,
                            n_positions = 1024,
                            n_embd = 768,
                            n_layer = 12,
                            n_head = 12,
                            use_bias=False)

    model = GPT2LMHeadModel(config)
    sd_hf = model.state_dict()
    k_hf = list(sd_hf.keys())

    sd_ngpt = torch.load(path)['model']

    unwanted_prefix = '_orig_mod.'
    for k,v in list(sd_ngpt.items()):
        if k.startswith(unwanted_prefix):
            sd_ngpt[k[len(unwanted_prefix):]] = sd_ngpt.pop(k)

    k_ngpt = list(sd_ngpt.keys())

    a = set(k_hf) - set(k_ngpt) 
    a = [i for i in a if 'bias' not in i]

    transposed = ['attn.c_attn.weight', 
                  'attn.c_proj.weight', 
                  'mlp.c_fc.weight', 
                  'mlp.c_proj.weight']
    
    for k in k_ngpt:
        if any(k.endswith(w) for w in transposed):
            assert sd_hf[k].shape[::-1] == sd_ngpt[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd_ngpt[k].t())
        else:
            assert sd_hf[k].shape == sd_ngpt[k].shape
            with torch.no_grad():
                sd_hf[k].copy_(sd_ngpt[k])
    
    model.to(device)
    return model
