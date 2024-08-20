import torch
from transformers import GPT2LMHeadModel, GPT2Config

# Inverse of : https://github.com/karpathy/nanoGPT/blob/9755682b981a45507f6eb9b11eadef8cb83cebd5/model.py#L207
# Converts a saved nanogpt model back to HF format so it can be loaded in vLLM

def convert_to_hf(path, device: str = 'cpu'):
    custom_gpt = torch.load(path, map_location=device)['model']

    # Update custom weights to match with GPT2 on HF
    # 1. Remove unwanted prefix
    # 2. Transpose certain layers

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
        activation_function='gelu'
    )

    model_args.update(custom_gpt_config)

    config = GPT2Config(**model_args)
    model = GPT2LMHeadModel(config)
    model.load_state_dict(clean_custom_gpt, strict=False)

    return model

if __name__ == '__main__':
    from argparse import ArgumentParser
    from tts.gpt2_model import get_model

    parser = ArgumentParser()
    parser.add_argument('--path', required=True, type=str)
    args = parser.parse_args()

    custom_model = get_model(
        vocab_size=53376,
        device='cpu',
        compile=False,
        path=args.path
    )

    custom_model.eval()

    model = convert_to_hf(args.path)
    model.eval()

    store_hf: dict = {}
    store_custom: dict = {}

    def hook(module, input, output, name, store):
        store[name] = output

    def register_hook(m, store):
        for name, layer in m.named_modules():
            layer.register_forward_hook(lambda layer, input, output, name=name: hook(
                layer, input, output, name, store))

    register_hook(model, store_hf)
    register_hook(custom_model, store_custom)

    inputs = torch.randint(0, 50000, (1, 100))

    with torch.no_grad():
        pretrained_out = model(inputs)
        custom_out = custom_model(inputs)

    for k, v in store_hf.items():
        if not k:
            continue

        if k not in store_custom:
            print(f'{k} not found in custom')

        else:
            val1 = v

            if k == 'lm_head':
                val1 = v[:, -1, :]
            if type(val1) == tuple:
                    val1 = v[0]

            val2 = store_custom[k]
            if type(val2) == tuple:
                val2 = val2[0]

            diff = val1 - val2
            diff = diff.abs()
            diff = diff.max()

            print(f'Diff for layer: {k}: {diff}')
