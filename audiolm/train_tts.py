import glob
import torch
import random

import numpy as np
from pathlib import Path
from dataclasses import dataclass

from gpt2_trainer import train as gpt_train
from gpt2_model import GPT
from tokenlib import TEXT, SEMANTIC, ACOUSTIC, AUDIO

DEVICE = 'cuda:0'

coarse_codebooks = 2
per_codebook_size = 1024

VOCAB_SIZES = {
    TEXT: 50257,
    SEMANTIC: 1000,
    ACOUSTIC: 2048,
}

OFFSET = {
    TEXT: 0,
    SEMANTIC: VOCAB_SIZES[TEXT],
    ACOUSTIC: VOCAB_SIZES[SEMANTIC],
}

PAD_TOKEN = {
    TEXT: 50256,
    SEMANTIC: OFFSET[SEMANTIC] + VOCAB_SIZES[SEMANTIC],
    ACOUSTIC: 3050,
}

class DataLoader:
    def __init__(self, data_dir, source, target, max_source_tokens=256):
        self.data_dir = data_dir
        self.source = source
        self.target = target
        self.files, self.filenames = self.load_files()
        self.max_source_tokens = max_source_tokens

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
        if self.source == self.target:
            return self.load_batch_lm(split, block_size, batch_size)
        else:
            return self.load_batch_trans(split, block_size, batch_size)

    def load_batch_lm(self, split, block_size, batch_size):
        target = self.target

        some_filenames = random.sample(self.filenames[split], batch_size)
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + PAD_TOKEN[target]
        y = y + PAD_TOKEN[target]

        for i in range(batch_size):
            f = some_filenames[i]
            target_arr = np.load(self.files[target][f])
            target_arr = target_arr[0:self.max_source_tokens]
            target_arr = target_arr + OFFSET[target]

            _x = target_arr[:block_size]
            _y = target_arr[1:block_size + 1]
            x[i][:len(_x)] = _x
            y[i][:len(_y)] = _y

        return x, y

    def load_batch_trans(self, split, block_size, batch_size):
        source = self.source
        target = self.target

        some_filenames = random.sample(self.filenames[split], batch_size)
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + PAD_TOKEN[target]
        y = y + PAD_TOKEN[target]

        for i in range(batch_size):
            f = some_filenames[i]
            source_arr = np.load(self.files[source][f]) + OFFSET[source]
            source_arr = source_arr[0: self.max_source_tokens]
            source_arr = np.append(source_arr, PAD_TOKEN[source])

            target_arr = np.load(self.files[target][f])
            target_arr = target_arr + OFFSET[target]

            if target == ACOUSTIC:
                target_arr = target_arr[:coarse_codebooks]  # pick only top codebooks
                target_arr = self.codebook_encoding(target_arr, per_codebook_size)

            target_arr = np.append(target_arr, PAD_TOKEN[target])

            tokens = np.hstack([source_arr, target_arr])
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

def get_vocab_size(source, target):
    vocab_size = max(OFFSET[source] + VOCAB_SIZES[source],
                    OFFSET[target] + VOCAB_SIZES[target],
                    PAD_TOKEN[source], PAD_TOKEN[target]) + 1

    return vocab_size

def load_model(expanded_vocab_size=None, weights_path=None):
    model = GPT.from_pretrained('cmeraki/gpt2-124M-400B')

    if expanded_vocab_size:
        model.expand_vocab(new_vocab_size=expanded_vocab_size)

    if weights_path:
        saved_model = torch.load(weights_path)['model']
        saved_model = {k.replace('_orig_mod.', ''): saved_model[k] for k in saved_model}

        model.load_state_dict(saved_model)

    model.to(DEVICE)

    model = torch.compile(model)

    return model

def train_translator(source, target, data_dir, model, out_dir):
    print(f"Training {source} {target}".upper())

    data_generator = DataLoader(data_dir=data_dir,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=15000,
              block_size=1024,
              eval_interval=500,
              eval_steps=100,
              batch_size=32,
              grad_accum_steps=4,
              device=DEVICE)

    return out_dir

def train():
    out_dir = Path('out_400b_ft_xs')
    data_dir = iter_dataset()

    source, target = TEXT, SEMANTIC
    vocab_size = get_vocab_size(source, target)
    print(f"{source}:{target} Vocab size", vocab_size)

    text_semantic_model = load_model(vocab_size)
    text_semantic_model_path = train_translator(TEXT,
                                                SEMANTIC,
                                                model=text_semantic_model,
                                                data_dir=data_dir,
                                                out_dir=out_dir)

    source, target = SEMANTIC, ACOUSTIC
    vocab_size = get_vocab_size(source, target)
    print(f"{source}:{target} Vocab size", vocab_size)

    text_semantic_model = load_model(vocab_size)
    text_semantic_model_path = train_translator(TEXT,
                                                SEMANTIC,
                                                model=text_semantic_model,
                                                data_dir=data_dir,
                                                out_dir=out_dir)


if __name__ == '__main__':
    train()