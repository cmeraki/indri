import numpy as np
import glob
import torch
import random

from pathlib import Path

data_dir = '../data/audio_tokens/gigaspeech_s/'

semantic_files = {Path(f).name: Path(f) for f in glob.glob(f"{data_dir}/semantic/*.npy")}
acoustic_files = {Path(f).name: Path(f) for f in glob.glob(f"{data_dir}/acoustic/*.npy")}
filenames = list(set(semantic_files).intersection(acoustic_files))

filenames = {
    'train': filenames[1000:],
    'val': filenames[:1000]
}

coarse_codebooks = 2
per_codebook_size = 1024

semantic_vocab_size = 1000
acoustic_vocab_size = coarse_codebooks * per_codebook_size
total_vocab_size = semantic_vocab_size + acoustic_vocab_size

semantic_pad_token = total_vocab_size + 1
acoustic_pad_token = total_vocab_size + 2

def codebook_encoding(arr: torch.tensor, per_codebook_size: int):
    c, n = arr.shape
    i_values = np.arange(c) * per_codebook_size
    arr += i_values.reshape(c, 1)
    flat_arr = arr.reshape(c*n, order='F')
    return flat_arr


def load_batch(split, batch_size, block_size):
    n_semantic_tokens = 256
    # remove 1 because pad tokens will be appended

    some_filenames = random.sample(filenames[split], batch_size)
    x = np.zeros(shape=(batch_size, block_size), dtype=np.int64) + acoustic_pad_token
    y = np.zeros(shape=(batch_size, block_size), dtype=np.int64) + acoustic_pad_token

    for i in range(batch_size):
        f = some_filenames[i]
        semantic = np.load(semantic_files[f]) + acoustic_vocab_size
        acoustic = np.load(acoustic_files[f])[:coarse_codebooks, :]

        semantic_start_position = 0
        semantic_end_position = semantic_start_position + n_semantic_tokens

        ratio = acoustic.shape[1] / semantic.shape[0]
        acoustic_start_position = int(ratio * semantic_start_position)
        acoustic_end_position = int(ratio * semantic_end_position)

        semantic = semantic[semantic_start_position: semantic_end_position]
        acoustic = acoustic[:, acoustic_start_position: acoustic_end_position]

        acoustic = codebook_encoding(acoustic, per_codebook_size)
        semantic = np.append(semantic, semantic_pad_token)
        acoustic = np.append(acoustic, acoustic_pad_token)

        tokens = np.hstack([semantic, acoustic])
        _x = tokens[:block_size]
        _y = tokens[1:block_size+1]
        x[i][:len(_x)] = _x
        y[i][:len(_y)] = _y

    return x, y


def get_batch(split, device, block_size, batch_size):
    device_type = 'cuda:0'

    x, y = load_batch(split, batch_size, block_size)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# from tqdm import tqdm
# for i in tqdm(range(1000)):
#     x, y = get_batch(split='train', device='cuda:0', block_size=1024, batch_size=64*16)
#     print(x.shape)
#
# # print(batch)