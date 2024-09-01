import glob
import torch
import random

import numpy as np
from pathlib import Path

from tts.gpt2_trainer import train as gpt_train
from tts.gpt2_model import get_model, GPT
from common import DEVICE
from common import Config as cfg
import json
from common import TEXT, SEMANTIC

print(cfg.__dict__)


def decorate(tokens, type):
    tokens = tokens + cfg.OFFSET[type]
    tokens = np.hstack([cfg.INFER_TOKEN[type],
                        tokens,
                        cfg.STOP_TOKEN[type]])
    return tokens


def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]


class DataLoader:
    def __init__(self, interleaved_dir, speech_dir, text_dir):
        self.interleaved_dir = interleaved_dir
        self.speech_dir = speech_dir
        self.text_dir = text_dir

        self.interleaved_files = self.load_interleaved(self.interleaved_dir)
        self.speech_files = self.load_numpy(self.speech_dir)
        self.text_files = self.load_numpy(self.text_dir)

    def load_numpy(self, data_dir):
        files = glob.glob(f'{data_dir}/*.npy')

        files = list(files)

        print("Num files", len(files))

        files = {
            'train': files[1000:],
            'val': files[:1000]
        }

        return files

    def load_interleaved(self, data_dir):
        files = glob.glob(f'{data_dir}/*.json')

        files = list(files)

        print("Num files", len(files))

        files = {
            'train': files[1000:],
            'val': files[:1000]
        }

        return files

    def get_tokens_interleaved(self, split):
        filename = random.choice(self.interleaved_files[split])
        data = json.load(open(filename))
        size = len(data[TEXT])
        modalities = [TEXT, SEMANTIC]
        choices = [random.randint(0, 1) for _ in range(size)]

        output = []
        for idx in range(size):
            choice = choices[idx]
            modality = modalities[choice]
            tokens = np.asarray(data[modality][idx])
            tokens = decorate(tokens, modality)
            output.extend(tokens)

        output = np.hstack(output)
        # print(output)
        return output

    def get_tokens_text(self, split):
        filename = random.choice(self.text_files[split])
        tokens = np.load(filename)
        tokens = decorate(tokens, TEXT)
        return tokens

    def get_tokens_speech(self, split):
        filename = random.choice(self.speech_files[split])
        tokens = np.load(filename)
        tokens = tokens.reshape(-1)
        tokens = replace_consecutive(tokens)
        tokens = decorate(tokens, SEMANTIC)
        return tokens

    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + cfg.OMNI_STOP_TOKEN
        y = y + cfg.OMNI_STOP_TOKEN
        methods = [self.get_tokens_interleaved,
                   self.get_tokens_speech,
                   self.get_tokens_text]

        for i in range(batch_size):
            method = random.choice(methods)
            tokens = method(split)
            index = random.randint(0, max(len(tokens) - block_size, 0))

            _x = tokens[index:block_size]
            _y = tokens[index + 1:block_size + 1]

            # print(method, _x.shape)

            x[i][:len(_x)] = _x
            y[i][:len(_y)] = _y

        return x, y

    def get_batch(self, split, device, block_size, batch_size):
        x, y = self.load_batch(split,
                               block_size=block_size,
                               batch_size=batch_size)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if 'cuda' in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y


def train_omni():
    from common import cache_dir

    # download_dataset(data_dir)
    out_dir = Path(f'{cache_dir}/data/models/omni_mixed_large/')

    interleaved_dir = f'{cache_dir}/tinystories_omni/'
    speech_dir = f'{cache_dir}mls_eng_10k/tokens/semantic/'
    text_dir = f'{cache_dir}/mls_eng_10k/tokens/text/'

    pretrained = 'mdouglas/llmc-gpt2-774M-150B'
    vocab_size = cfg.VOCAB_SIZE

    model = get_model(path=pretrained)
    # model.expand_vocab(new_vocab_size=vocab_size)
    model.to(DEVICE)

    model = torch.compile(model)

    print(model)

    print("Vocab size", vocab_size)
    print("Model outdir", out_dir)
    print("Training omni".upper())

    data_generator = DataLoader(interleaved_dir=interleaved_dir,
                                speech_dir=speech_dir,
                                text_dir=text_dir)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=10000,
              block_size=1024,
              eval_interval=100,
              batch_size=8,
              grad_accum_steps=8,
              device=DEVICE)

    return out_dir


def train():
    from common import cache_dir

    interleaved_dir = f'{cache_dir}/tinystories_omni/'
    speech_dir = f'{cache_dir}mls_eng_10k/tokens/semantic/'
    text_dir = f'{cache_dir}/mls_eng_10k/tokens/text/'

    dl = DataLoader(interleaved_dir=interleaved_dir, speech_dir=speech_dir, text_dir=text_dir)
    for i in range(10):
        batch = dl.get_batch('train', DEVICE, 1024, 1)


if __name__ == '__main__':
    train_omni()