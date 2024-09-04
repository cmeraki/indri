import json
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
from omni.logger import get_logger
from datalib.tokenlib import get_tokenizer

logger = get_logger(__name__)
logger.info(cfg.__dict__)

def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]

class DataLoader:
    def __init__(
        self,
        data_dirs,
        max_source_tokens=256,
        prompt_length=0,
        max_files=None
    ):

        self.data_dirs = data_dirs
        self.max_files = max_files
        self.max_source_tokens = max_source_tokens
        self.prompt_length = prompt_length
        self.prompting = False
        self.text_tokenizer = get_tokenizer(type=TEXT, device='cpu')

        for idx in range(cfg.VOCAB_SIZES[SEMANTIC]):
            self.text_tokenizer.tokenizer.add_tokens(f'[sem_{idx}]')

        for idx in range(cfg.VOCAB_SIZES[ACOUSTIC]):
            self.text_tokenizer.tokenizer.add_tokens(f'[aco_{idx}]')

        for tok in list(cfg.MODALITY_TOKENS.values()) + list(cfg.TASK_TOKENS.values()) + [cfg.STOP_TOKEN]:
            print('Adding token: ', tok)
            self.text_tokenizer.tokenizer.add_tokens(tok)

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.continue_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONTINUE])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.semantic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[SEMANTIC])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[ACOUSTIC])

        print(f'Training semantic acoustic model with max source tokens {max_source_tokens} and max files {max_files}')

        self.load_parallel_data(self.data_dirs)

    def speaker_id_to_text(self, id, dataset=None):
        if dataset:
            text = f'[spkr_{dataset}_{id}]'
        else:
            if id is not None and len(id) > 0:
                text = f'[spkr_{id}]'
            else:
                text = '[spkr_unk]'
        
        return self.text_tokenizer.encode(text)

    def load_parallel_data(self, dirs):
        metadata = {}

        for dir in dirs:
            metadata_path = Path(dir, 'annotation', 'metadata.jsonl')

            for line in open(metadata_path):
                _metadata = json.loads(line.strip())
                _metadata['dir'] = dir
                metadata[_metadata['id']] = _metadata

        print("Num metadata lines: ", len(metadata))

        self.ids = list(metadata.keys())
        random.shuffle(self.ids)
        self.ids = {
            'train' : self.ids[1000:],
            'val' : self.ids[:1000],
        }

        self.metadata = metadata

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

    def get_tokens_path(self, id, type):
        sample = self.metadata[id]
        return Path(sample['dir'], sample[f'{type}_tokens'].replace('.opus', '').replace('.flac', ''))

    def get_semantic_acoustic_translation(self, split):
        id = random.choice(self.ids[split])

        source_arr = np.load(self.get_tokens_path(id, SEMANTIC)).astype(np.int64)
        source_arr = source_arr + cfg.OFFSET[SEMANTIC]
        source_arr = np.reshape(source_arr, -1)
        source_arr = replace_consecutive(source_arr)

        target_arr = np.load(self.get_tokens_path(id, ACOUSTIC)).astype(np.int64)
        target_arr = target_arr[:cfg.coarse_codebooks]
        target_arr = DataLoader.codebook_encoding(target_arr, cfg.per_codebook_size)
        target_arr = np.reshape(target_arr, -1)
        target_arr = target_arr + cfg.OFFSET[ACOUSTIC]

        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)
        null_speaker_id = self.speaker_id_to_text(None)
        speaker_tokens = speaker_id
        if np.random.random() < 0.2:
            speaker_tokens = null_speaker_id

        return np.hstack([
            self.semantic_modality_token,
            speaker_tokens,
            source_arr,
            self.convert_token,
            self.acoustic_modality_token,
            speaker_tokens,
            target_arr,
            self.stop_token
        ])


    def get_acoustic_semantic_translation(self, split):
        id = random.choice(self.ids[split])

        source_arr = np.load(self.get_tokens_path(id, ACOUSTIC)).astype(np.int64)
        source_arr = source_arr[:cfg.coarse_codebooks]
        source_arr = DataLoader.codebook_encoding(source_arr, cfg.per_codebook_size)
        source_arr = np.reshape(source_arr, -1)
        source_arr = source_arr + cfg.OFFSET[ACOUSTIC]

        target_arr = np.load(self.get_tokens_path(id, SEMANTIC)).astype(np.int64)
        target_arr = target_arr + cfg.OFFSET[SEMANTIC]
        target_arr = np.reshape(target_arr, -1)
        target_arr = replace_consecutive(target_arr)
        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)
        null_speaker_id = self.speaker_id_to_text(None)
        speaker_tokens = speaker_id
        if np.random.random() < 0.2:
            speaker_tokens = null_speaker_id

        return np.hstack([
            self.acoustic_modality_token,
            speaker_tokens,
            source_arr,
            self.convert_token,
            self.semantic_modality_token,
            speaker_tokens,
            target_arr,
            self.stop_token
        ])

    def get_semantic_acoustic_continue(self, split):
        pass

    def get_acoustic_semantic_continue(self, split):
        pass

    def get_acoustic_acoustic(self, split):
        id = random.choice(self.ids[split])

        source_arr = np.load(self.get_tokens_path(id, ACOUSTIC)).astype(np.int64)
        source_arr = source_arr[:cfg.coarse_codebooks]
        source_arr = DataLoader.codebook_encoding(source_arr, cfg.per_codebook_size)
        source_arr = np.reshape(source_arr, -1)
        source_arr = source_arr + cfg.OFFSET[ACOUSTIC]

        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)
        null_speaker_id = self.speaker_id_to_text(None)
        speaker_tokens = speaker_id
        if np.random.random() < 0.2:
            speaker_tokens = null_speaker_id

        return np.hstack([
            self.acoustic_modality_token,
            speaker_tokens,
            source_arr,
            self.stop_token
        ])

    def get_semantic_semantic(self, split):
        id = random.choice(self.ids[split])

        source_arr = np.load(self.get_tokens_path(id, SEMANTIC)).astype(np.int64)
        source_arr = source_arr + cfg.OFFSET[SEMANTIC]
        source_arr = np.reshape(source_arr, -1)
        source_arr = replace_consecutive(source_arr)

        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)
        null_speaker_id = self.speaker_id_to_text(None)
        speaker_tokens = speaker_id
        if np.random.random() < 0.2:
            speaker_tokens = null_speaker_id

        return np.hstack([
            self.semantic_modality_token,
            speaker_tokens,
            source_arr,
            self.stop_token
        ])

    def get_valid_sequence(self, split):
        task_methods = [
            self.get_semantic_acoustic_translation,
            self.get_acoustic_semantic_translation,
            self.get_acoustic_acoustic,
            self.get_semantic_semantic,
        ]

        while True:
            try:
                method = random.choice(task_methods)
                yield method(split), method.__name__
            except Exception as e:
                pass


    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + self.stop_token
        y = y + self.stop_token

        i = 0
        start_idx = 0
        end_idx = 0

        tasks = []

        for tokens, task in self.get_valid_sequence(split):
            tasks.append(task)
            new_toks = block_size - end_idx
            _x = tokens[:new_toks]
            _y = tokens[1:new_toks + 1]
            if len(_x) != len(_y):
                _y = np.hstack([_y, self.stop_token])

            end_idx = start_idx + len(_x)

            x[i][start_idx:end_idx] = _x
            y[i][start_idx:end_idx] = _y

            start_idx = end_idx

            if end_idx < block_size:
                continue

            i += 1
            start_idx = 0
            end_idx = 0

            if i == batch_size:
                break

        import pdb; pdb.set_trace()
        return x, y, tasks

    def get_batch(self, split, device, block_size, batch_size):
        x, y, tasks = self.load_batch(split, block_size=block_size, batch_size=batch_size)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if 'cuda' in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y, tasks

def train(args):
    from datetime import datetime

    today = datetime.today().strftime('%y%m%d-%H%M%S')
    out_dir = Path(f'{CACHE_DIR}/romit/models/semantic_acoustic_tasks_small')
    data_dirs = [
        Path(f'{CACHE_DIR}/mls_eng_10k'),
        Path(f'{CACHE_DIR}/data/peoples_speech_tokens'),
        Path(f'{CACHE_DIR}/data/gs_xl_en_tokens'),
        Path(f'{CACHE_DIR}/jenny'),
        Path(f'{CACHE_DIR}/hifi_tts'),
        Path(f'{CACHE_DIR}/expresso'),
    ]

    print("Data dirs: ", data_dirs)
    print("Out dir: ", out_dir)

    model = get_model(
        model_type=training_cfg.MODEL_TYPE,
        vocab_size=cfg.VOCAB_SIZE,
        block_size=training_cfg.BLOCK_SIZE,
        device=DEVICE,
    )

    logger.info(model)

    print(f"Training {training_cfg.MODEL_TYPE} semantic acoustic")
    print(f"Vocab size {cfg.VOCAB_SIZE}")

    data_generator = DataLoader(
        data_dirs=data_dirs,
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
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    train(args)
