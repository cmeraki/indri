import numpy as np
import glob
import torch
import random

from pathlib import Path
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, IMAGE

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
        flat_arr = arr.reshape(c*n, order='F')
        return flat_arr

    def load_batch(self, split, block_size, batch_size):
        if self.source == self.target:
            return  self.load_batch_lm(split, block_size, batch_size)
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
            _y = tokens[1:block_size+1]
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