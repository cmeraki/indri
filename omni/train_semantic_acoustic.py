import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import pdb

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR
from configs.constants import *
import configs.training_semantic_acoustic as training_cfg

from omni.gpt2_model import get_model
from omni.gpt2_trainer import train as gpt_train
from datalib.tokenlib import get_tokenizer

print(cfg.__dict__)

class DataLoader:
    def __init__(
        self,
        data_dir,
        max_source_tokens=256,
        prompt_length=0,
        max_files=None
    ):

        self.data_dir = data_dir
        self.max_files = max_files
        self.max_source_tokens = max_source_tokens
        self.prompt_length = prompt_length
        self.prompting = False
        self.types = [SEMANTIC, ACOUSTIC]

        self.files, self.filenames = self.load_files()

        self.text_tokenizer = get_tokenizer(type=TEXT, device='cpu')
        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.continue_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONTINUE])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.semantic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[SEMANTIC])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[ACOUSTIC])

        print(f'Training semantic acoustic model with max source tokens {max_source_tokens} and max files {max_files}')

    def load_files(self):
        files = {}
        filenames = None

        for type in [SEMANTIC, ACOUSTIC]:
            files[type] = {}
            file_iter = tqdm(glob.iglob(f"{self.data_dir}/{type}/**/*.npy", recursive=True), desc=f'reading {type}')
            for f in file_iter:
                files[type][Path(f).name] = Path(f)
                if self.max_files:
                    if len(files[type]) >= self.max_files:
                        break

            if not filenames:
                filenames = set(files[type].keys())

            filenames = filenames.intersection(set(files[type].keys()))

        filenames = list(filenames)

        print(f"Num files: {len(filenames)}")

        filenames = {
            'train': filenames[1000:],
            'val': filenames[:1000]
        }

        return files, filenames

    @staticmethod
    def codebook_encoding(
        arr: torch.tensor,
        per_codebook_size: int
    ):

        # Interleave n codebooks as 1
        c, n = arr.shape
        i_values = np.arange(c) * per_codebook_size
        arr += i_values.reshape(c, 1)
        flat_arr = arr.reshape(c * n, order='F')
        return flat_arr

    def get_semantic_acoustic_translation(self, split):
        filename = random.choice(self.filenames[split])

        source_arr = np.load(self.files[SEMANTIC][filename]).astype(np.int64)
        source_arr = source_arr + cfg.OFFSET[SEMANTIC]
        source_arr = np.reshape(source_arr, -1)

        target_arr = np.load(self.files[ACOUSTIC][filename]).astype(np.int64)
        target_arr = target_arr[:cfg.coarse_codebooks]
        target_arr = DataLoader.codebook_encoding(target_arr, cfg.per_codebook_size)
        target_arr = np.reshape(target_arr, -1)
        target_arr = target_arr + cfg.OFFSET[ACOUSTIC]

        return np.hstack([
            self.semantic_modality_token,
            source_arr,
            self.convert_token,
            self.acoustic_modality_token,
            target_arr
        ])


    def get_acoustic_semantic_translation(self, split):
        filename = random.choice(self.filenames[split])

        source_arr = np.load(self.files[ACOUSTIC][filename]).astype(np.int64)
        source_arr = source_arr[:cfg.coarse_codebooks]
        source_arr = DataLoader.codebook_encoding(source_arr, cfg.per_codebook_size)
        source_arr = np.reshape(source_arr, -1)
        source_arr = source_arr + cfg.OFFSET[ACOUSTIC]

        target_arr = np.load(self.files[SEMANTIC][filename]).astype(np.int64)
        target_arr = target_arr + cfg.OFFSET[SEMANTIC]
        target_arr = np.reshape(target_arr, -1)

        return np.hstack([
            self.acoustic_modality_token,
            source_arr,
            self.convert_token,
            self.semantic_modality_token,
            target_arr
        ])

    def get_semantic_acoustic_continue(self, split):
        pass

    def get_acoustic_semantic_continue(self, split):
        pass

    def get_acoustic_acoustic(self, split):
        filename = random.choice(self.filenames[split])

        source_arr = np.load(self.files[ACOUSTIC][filename]).astype(np.int64)
        source_arr = source_arr[:cfg.coarse_codebooks]
        source_arr = DataLoader.codebook_encoding(source_arr, cfg.per_codebook_size)
        source_arr = np.reshape(source_arr, -1)
        source_arr = source_arr + cfg.OFFSET[ACOUSTIC]

        random_cut = random.randint(0, len(source_arr) - 1)

        return np.hstack([
            self.acoustic_modality_token,
            source_arr[:random_cut],
            self.continue_token,
            self.acoustic_modality_token,
            source_arr[random_cut + 1:],
        ])

    def get_semantic_semantic(self, split):
        filename = random.choice(self.filenames[split])

        source_arr = np.load(self.files[SEMANTIC][filename]).astype(np.int64)
        source_arr = source_arr + cfg.OFFSET[SEMANTIC]
        source_arr = np.reshape(source_arr, -1)

        random_cut = random.randint(0, len(source_arr) - 1)

        return np.hstack([
            self.semantic_modality_token,
            source_arr[:random_cut],
            self.continue_token,
            self.semantic_modality_token,
            source_arr[random_cut + 1:],
        ])

    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + self.stop_token
        y = y + self.stop_token

        task_methods = [
            self.get_semantic_acoustic_translation,
            self.get_acoustic_semantic_translation,
            self.get_acoustic_acoustic,
            self.get_semantic_semantic,
        ]

        for i in range(batch_size):
            method = random.choice(task_methods)
            # pdb.set_trace()
            tokens = method(split)
            # pdb.set_trace()

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

def train():
    from datetime import datetime

    today = datetime.today().strftime('%y%m%d-%H%M%S')
    data_dir = Path(os.path.join(CACHE_DIR, 'romit', 'data'))
    out_dir = Path(os.path.join(CACHE_DIR, 'romit', 'models', today, 'semantic_acoustic'))

    print("Data dir: ", data_dir)
    print("Out dir: ", out_dir)

    model = get_model(
        model_type=training_cfg.MODEL_TYPE,
        vocab_size=cfg.VOCAB_SIZE,
        block_size=training_cfg.BLOCK_SIZE,
        device=DEVICE,
    )

    print(f"Training {training_cfg.MODEL_TYPE} semantic acoustic")
    print(f"Vocab size {cfg.VOCAB_SIZE}")

    data_generator = DataLoader(
        data_dir=data_dir,
        max_source_tokens=training_cfg.MAX_SOURCE_TOKENS,
    )

    gpt_train(
        model,
        get_batch=data_generator.get_batch,
        out_dir=out_dir,
        steps=training_cfg.STEPS,
        block_size=training_cfg.BLOCK_SIZE,
        eval_interval=training_cfg.EVAL_INTERVAL,
        eval_steps=training_cfg.EVAL_STEPS,
        batch_size=training_cfg.BATCH_SIZE,
        grad_accum_steps=training_cfg.GRAD_ACCUM_STEPS,
        device=DEVICE
    )


if __name__ == '__main__':
    train()
