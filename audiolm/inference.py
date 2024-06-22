import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"

import numpy as np
import bark
import torch
from encodec.utils import save_audio
from datalib import semantic_pad_token, acoustic_pad_token, acoustic_vocab_size, semantic_vocab_size
from audio_tokenizers import EncodecTokenizer

def load_llm():
    from gpt2_model import GPT, GPTConfig
    from contextlib import nullcontext
    import os
    from tqdm import tqdm

    num_samples = 10 # number of samples to draw
    max_new_tokens = 1024 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cuda:0' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    compile = False # use PyTorch 2.0 to compile the model to be faster

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print("Loading model")
    # model
    # init from a model saved in a specific directory
    ckpt_path = os.path.join('out', 'ckpt_2048.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    print("Loaded model")
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # print(model)

    print("Acoustic pad", acoustic_pad_token)
    print("Semantic pad", semantic_pad_token)

    start_ids = np.load('../data/audio_tokens/semantic/AUD0000000468_S0000008.wav.npy')
    start_ids = start_ids[:250] + acoustic_vocab_size
    start_ids = np.append(start_ids, semantic_pad_token)
    print(start_ids)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    model.eval()

    tokenizer = EncodecTokenizer()

    # run generation
    with torch.no_grad():
        with ctx:
            for k in tqdm(range(num_samples)):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                y = y.detach().cpu().numpy()[0]
                print(list(y))
                start_idx = np.where(y == semantic_pad_token)[0][0]
                end_idx = np.where(y == acoustic_pad_token)[0][0]
                y = y[start_idx + 1: end_idx]
                print(y)
                wav = tokenizer.decode(y)

                save_audio(wav[0], f'{k}.wav', sample_rate=24000)

if __name__ == "__main__":
    # tokens = np.load('data/audio_tokens/74de910e-e10c-418b-8c22-11ab58e5cd13.npy')
    # print(tokens.shape)
    # test_generation(tokens)
    load_llm()