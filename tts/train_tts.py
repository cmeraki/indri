import glob
import torch
import random

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from gpt2_trainer import train as gpt_train
from gpt2_model import GPT, get_model
from tokenlib import TEXT, SEMANTIC, ACOUSTIC, AUDIO

DEVICE = 'cuda:0'

coarse_codebooks = 2
per_codebook_size = 1024

class cfg:
    VOCAB_SIZES = {
        TEXT: 50257,
        SEMANTIC: 1000,
        ACOUSTIC: 2048,
    }

    OFFSET = {
        TEXT: 0,
        SEMANTIC: VOCAB_SIZES[TEXT],
        ACOUSTIC: VOCAB_SIZES[TEXT] + VOCAB_SIZES[SEMANTIC],
    }
    
    max_token_value = 0
    for i in OFFSET:
        max_token_value = max(OFFSET[i] + VOCAB_SIZES[i], max_token_value)
        

    PAD_TOKEN = {
        TEXT: 50256,
        SEMANTIC: max_token_value + 2,
        ACOUSTIC: max_token_value + 3,
    }

    PROMPT_TOKEN = {
        TEXT: max_token_value + 4,
        SEMANTIC: max_token_value + 5,
        ACOUSTIC: max_token_value + 6,
    }

    VOCAB_SIZE = (max(PROMPT_TOKEN.values()) // 64 + 1)*64

class DataLoader:
    def __init__(self, data_dir, source, target, max_source_tokens=256, prompt_length=0):
        self.data_dir = data_dir
        self.source = source
        self.target = target
        self.files, self.filenames = self.load_files()
        self.max_source_tokens = max_source_tokens
        self.prompt_length = prompt_length

    def load_files(self):
        files = {}
        filenames = None

        for type in [self.source, self.target]:
            # create a dictionary of name: filepath mapping
            files[type] = {Path(f).name: Path(f) for f in glob.glob(f"{self.data_dir}/{type}/*.npy")}

            if not filenames:
                filenames = set(files[type].keys())

            filenames = filenames.intersection(set(files[type].keys()))

        filenames = list(filenames)

        filenames = {
            'train': filenames[1000:],
            'val': filenames[:1000]
        }

        return files, filenames

    @staticmethod
    def codebook_encoding(arr: torch.tensor,
                          per_codebook_size: int):

        # interleave n codebooks as 1
        c, n = arr.shape
        i_values = np.arange(c) * per_codebook_size
        arr += i_values.reshape(c, 1)
        flat_arr = arr.reshape(c * n, order='F')
        return flat_arr

    def load_batch(self, split, block_size, batch_size):
        source = self.source
        target = self.target

        some_filenames = random.sample(self.filenames[split], batch_size)
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + cfg.PAD_TOKEN[target]
        y = y + cfg.PAD_TOKEN[target]

        for i in range(batch_size):
            f = some_filenames[i]
            source_arr = np.load(self.files[source][f]) + cfg.OFFSET[source]
            source_arr = source_arr[0: self.max_source_tokens]
            source_arr = np.append(source_arr, cfg.PAD_TOKEN[source])

            target_arr = np.load(self.files[target][f])
            target_arr = target_arr + cfg.OFFSET[target]

            if target == ACOUSTIC:
                target_arr = target_arr[:coarse_codebooks]  # pick only top codebooks
                target_arr = self.codebook_encoding(target_arr, per_codebook_size)
            
            prompt_arr = np.asarray([cfg.PROMPT_TOKEN[target]])

            if self.prompt_length > 0:
                prompt_idx_start = np.random.randint(0, len(target_arr) - self.prompt_length + 1)
                prompt_arr = target_arr[prompt_idx_start : prompt_idx_start + self.prompt_length]
                prompt_arr = np.append(prompt_arr, cfg.PROMPT_TOKEN[target])

            target_arr = np.append(target_arr, cfg.PAD_TOKEN[target])
            
            tokens = np.hstack([source_arr, prompt_arr, target_arr])
            _x = tokens[:block_size]
            _y = tokens[1:block_size + 1]
            x[i][:len(_x)] = _x
            y[i][:len(_y)] = _y

        return x, y

    def get_batch(self, split, device, block_size, batch_size):
        x, y = self.load_batch(split, block_size=block_size, batch_size=batch_size)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if 'cuda' in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y

def get_vocab_size():
    vocab_size = cfg.VOCAB_SIZE

    return vocab_size

def train_translator(source, target, data_dir, out_dir, prompt_length=0):
    vocab_size = cfg.VOCAB_SIZE
    print(f"{source}:{target} Vocab size", vocab_size)

    model = get_model(vocab_size=vocab_size, device=DEVICE)

    print(f"Training {source} {target}".upper())

    data_generator = DataLoader(data_dir=data_dir,
                                source=source,
                                target=target,
                                prompt_length=prompt_length)


    out_dir = out_dir / f'{source}_{target}'

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=1000,
              block_size=1024,
              eval_interval=100,
              eval_steps=10,
              batch_size=40,
              grad_accum_steps=4,
              device=DEVICE)

    return out_dir

def train():
    data_dir = '/home/apurva/projects/indri/data/speechcolab/gigaspeech/'
    out_dir = Path('out_400b_ft_xs')
    train_translator(TEXT, SEMANTIC, data_dir, out_dir, prompt_length=25)
    train_translator(SEMANTIC, ACOUSTIC, data_dir, out_dir, prompt_length=64)


if __name__ == '__main__':
    train()