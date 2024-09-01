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
from tqdm import tqdm
from common import TEXT, SEMANTIC
from datalib.datalib import Dataset
from datalib.tokenlib import get_tokenizer

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


# Datasets
# TTS speaker datasets : hifi, cmu, expresso, jenny + mlseng
# TTS datasets : gs, ps, mlseng
# TTS cleanup required in text : gigaspeech
# mlseng need to recreate metadata
# Cartoon data scraping : reject data from onepiece, need to scrape a good set
# prepare annotations on gigaspeech, it was made long long ago

class DataLoader:
    def __init__(self, interleaved_dirs, datasets_dirs):
        # types of data on which training is done
        # TTS
        # TTS with speaker id
        # ASR
        # continuation
            # text-text
            # text-semantic
            # semantic-text
        # raw semantic
        # raw text

        self.interleaved_dirs = interleaved_dirs
        self.dataset_dirs = datasets_dirs

        self.text_tokenizer = get_tokenizer(type=TEXT, device='cpu')

        self.load_parallel_data(self.dataset_dirs)
        self.interleaved_files = self.load_interleaved(self.interleaved_dirs)


        self.tts_start_token = self.text_tokenizer.encode('[TTS]')
        self.asr_start_token = self.text_tokenizer.encode('[ASR]')
        self.continue_token = self.text_tokenizer.encode('[CONTINUE]')


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
            dir = Path(dir)
            metadata_path =  dir / 'annotation' / 'metadata.jsonl'
            for line in open(metadata_path):
                _metadata = json.loads(line.strip())
                _metadata['dir'] = dir
                metadata[_metadata['id']] = _metadata

        print("num metadata lines", len(metadata))

        self.ids = list(metadata.keys())
        
        self.ids = {
            'train' : self.ids[1000:],
            'val' : self.ids[:1000],
        }

        self.metadata = metadata

    def load_interleaved(self, data_dir):
        files = glob.glob(f'{data_dir}/*.json')
        files = list(files)

        print("Num interleaved files", len(files))

        files = {
            'train': files[1000:],
            'val': files[:1000]
        }

        return files

    def get_tokens_continue(self, split):
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
        return output

    def normalize_text(self, text):
        text = text.lower()
        text = text.replace("<comma>", ',')
        text = text.replace("<period>", '.')
        text = text.replace('<questionmark>', '?')
        text = text.replace('<exclamationpoint>', '!')
        text = text.replace("\n", " ")
        return text

    def get_text_tokens_for_id(self, id):
        sample = self.metadata[id]
        text = sample['raw_text']
        norm_text = self.normalize_text(text)
        tokens = np.asarray(self.text_tokenizer.encode(norm_text))
        return tokens

    def get_tokens_text(self, split):
        id = random.choice(self.ids[split])
        tokens = self.get_text_tokens_for_id(id)
        tokens = decorate(tokens, TEXT)
        return tokens

    def get_tokens_speech(self, split):
        id = random.choice(self.ids[split])
        filename = self.get_tokens_path(id, SEMANTIC)
        tokens = np.load(filename)
        tokens = tokens.reshape(-1)
        tokens = replace_consecutive(tokens)
        tokens = decorate(tokens, SEMANTIC)
        return tokens

    def get_tokens_path(self, id, type):
        sample = self.metadata[id]
        return sample['dir'] / sample[f'{type}_tokens']

    def get_tts(self, split):
        # changing the way loading happens for asr and tts
        # now that we have speaker ids, loading would start from metadata
        id = random.choice(self.ids[split])
        text_tokens = self.get_text_tokens_for_id(id) + cfg.OFFSET[TEXT]
        speech_tokens = np.load(self.get_tokens_path(id, SEMANTIC)).reshape(-1) + cfg.OFFSET[SEMANTIC]
        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)
        tokens = [text_tokens, self.tts_start_token, speaker_id, speech_tokens, cfg.STOP_TOKEN[SEMANTIC]]
        tokens = np.hstack(tokens)
        return tokens
    
    def get_asr(self, split):
        # changing the way loading happens for asr and tts
        # now that we have speaker ids, loading would start from metadata
        id = random.choice(self.ids[split])
        text_tokens = self.get_text_tokens_for_id(id) + cfg.OFFSET[TEXT]
        speech_tokens = np.load(self.get_tokens_path(id, SEMANTIC)).reshape(-1) + cfg.OFFSET[SEMANTIC]
        tokens = [speech_tokens, self.asr_start_token, text_tokens, cfg.STOP_TOKEN[TEXT]]
        tokens = np.hstack(tokens)
        return tokens

    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + cfg.OMNI_STOP_TOKEN
        y = y + cfg.OMNI_STOP_TOKEN

        methods = [self.get_tokens_speech,
                   self.get_tokens_text,
                   self.get_asr,
                   self.get_tts,
                   self.get_tokens_continue,
                   ]

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

    interleaved_dir = [f'{cache_dir}/tinystories_omni/']

    datasets_dirs = [f'{cache_dir}/jenny/',
                     f'{cache_dir}/expresso/', 
                     f'{cache_dir}/mls_eng_10k/', 
                    f'{cache_dir}/peoples_speech/', 
                    f'{cache_dir}/gs_xl_en_tokens/']
    

    datasets_dirs = [f'{cache_dir}/jenny/',
                      f'{cache_dir}/expresso/',]
    

    data_generator = DataLoader(interleaved_dirs=[interleaved_dir],
                                datasets_dirs=datasets_dirs)

    pretrained = 'mdouglas/llmc-gpt2-774M-150B'
    vocab_size = cfg.VOCAB_SIZE

    model = GPT.from_pretrained(model_type='cmeraki/gpt2-124M-400B')
    model.expand_vocab(new_vocab_size=vocab_size)
    model.to(DEVICE)

    model = torch.compile(model)

    print(model)

    print("Vocab size", vocab_size)
    print("Model outdir", out_dir)
    print("Training omni".upper())


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


if __name__ == '__main__':
    train_omni()