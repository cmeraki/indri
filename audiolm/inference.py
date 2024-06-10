import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"

from tokenize_audio import get_model, START_TOKEN
import numpy as np
import bark
import torch
from encodec.utils import save_audio


def deserialize_tokens(tokens: np.ndarray, n_codebooks=2):
    # serial token shape = n,1
    # deserialize to (codebook, tokens)
    # remove start_token
    start_indices = tokens == START_TOKEN
    start_indices = np.argwhere(start_indices).reshape(-1)
    start_indices = start_indices[1:]
    splits = np.split(tokens, indices_or_sections=start_indices)
    codebook_deindex = np.arange(n_codebooks) * 1024
    codebook_deindex = np.expand_dims(codebook_deindex, axis=-1)
    splits = [split[1:len(split)-1+len(split)%2].reshape((2, split[1:].shape[0] // 2), order='F') - codebook_deindex for split in splits]
    return splits


def decode_to_audio(tokens):
    model = get_model(bandwidth=3)
    tokens = deserialize_tokens(tokens)
    print(tokens)
    token_single = np.expand_dims(tokens[0], axis=0)
    good_audio = bark.api.generate_fine(x_coarse_gen=token_single[0, 0:2, :], silent=False)
    good_audio = np.expand_dims(good_audio, axis=0)
    good_audio = torch.from_numpy(good_audio)
    wav = model.decode([(good_audio, None)])

    return wav

def load_llm():
    from gpt2_model import GPT, GPTConfig
    from contextlib import nullcontext
    import os
    from tqdm import tqdm

    num_samples = 10 # number of samples to draw
    max_new_tokens = 512 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 100 # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed = 1337
    device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
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
    ckpt_path = os.path.join('out', 'ckpt.pt')
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
    # start_ids = np.load('data/audio_tokens/05d3b18a-0986-4087-8cf6-ced997470ad2.npy')[:64]
    start_ids = [0, 408]
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
    model.eval()
    # run generation
    with torch.no_grad():
        with ctx:
            for k in tqdm(range(num_samples)):
                # try:
                    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                    y = y.detach().cpu().numpy()[0]
                    wav = decode_to_audio(y)
                    save_audio(wav[0], f'{k}.wav', sample_rate=24000)
                    # start_ids = y[0][-101:]
                    print('I made it')
                # except:
                #     print("I have failed")

if __name__ == "__main__":
    # tokens = np.load('data/audio_tokens/74de910e-e10c-418b-8c22-11ab58e5cd13.npy')
    # print(tokens.shape)
    # test_generation(tokens)
    load_llm()