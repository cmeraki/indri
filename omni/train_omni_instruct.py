import json
import torch
import random
import numpy as np
from pathlib import Path

from datalib.tokenlib import get_tokenizer
from omni.gpt2_trainer import train as gpt_train
from omni.gpt2_model import GPT

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR
from configs.constants import *
import configs.training_omni as training_cfg


print(cfg.__dict__)

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

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.continue_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONTINUE])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.semantic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[SEMANTIC])
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])


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

    def load_interleaved(self, data_dir: list[Path]):

        files = []
        for dir in data_dir:
            files.extend(list(dir.glob('**/*.json')))

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
        # for idx in range(size):
        #     choice = choices[idx]
        #     modality = modalities[choice]
        #     tokens = np.asarray(data[modality][idx])
        #     tokens = decorate(tokens, modality)
        #     output.extend(tokens)

        for idx in range(size):
            choice = choices[idx]
            modality = modalities[choice]
            tokens = np.asarray(data[modality][idx])
            tokens = tokens + cfg.OFFSET[modality]
            if idx == 0 and modality == TEXT:
                tokens = np.hstack([self.text_modality_token, tokens])
            elif idx == 0 and modality == SEMANTIC:
                tokens = np.hstack([self.semantic_modality_token, tokens])
            elif modality == TEXT:
                tokens = np.hstack([self.continue_token, self.text_modality_token, tokens])
            elif modality == SEMANTIC:
                tokens = np.hstack([self.continue_token, self.semantic_modality_token, tokens])
            output.extend(tokens)

        output = np.hstack(output, self.stop_token)
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
        tokens = tokens + cfg.OFFSET[TEXT]

        random_cut = random.randint(0, len(tokens) - 1)
        tokens = np.hstack([
            self.text_modality_token,
            tokens[:random_cut],
            self.continue_token,
            self.text_modality_token,
            tokens[random_cut:],
            self.stop_token
        ])
        return tokens

    def get_tokens_speech(self, split):
        id = random.choice(self.ids[split])
        filename = self.get_tokens_path(id, SEMANTIC)
        tokens = np.load(filename)
        tokens = tokens.reshape(-1)
        tokens = replace_consecutive(tokens)
        tokens = tokens + cfg.OFFSET[SEMANTIC]

        speaker_id = self.metadata[id]['speaker_id']
        speaker_id = self.speaker_id_to_text(speaker_id)

        random_cut = random.randint(0, len(tokens) - 1)
        tokens = np.hstack([
            self.semantic_modality_token,
            tokens[:random_cut],
            self.continue_token,
            speaker_id,
            self.semantic_modality_token,
            tokens[random_cut:],
            self.stop_token
        ])
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
        speech_tokens = replace_consecutive(speech_tokens)

        tokens = np.hstack([
            self.text_modality_token,
            text_tokens,
            self.convert_token,
            speaker_id,
            self.semantic_modality_token,
            speech_tokens,
            self.stop_token
        ])

        return tokens
    
    def get_asr(self, split):
        # changing the way loading happens for asr and tts
        # now that we have speaker ids, loading would start from metadata
        id = random.choice(self.ids[split])
        text_tokens = self.get_text_tokens_for_id(id) + cfg.OFFSET[TEXT]
        speech_tokens = np.load(self.get_tokens_path(id, SEMANTIC)).reshape(-1) + cfg.OFFSET[SEMANTIC]
        speech_tokens = replace_consecutive(speech_tokens)

        tokens = np.hstack([
            self.semantic_modality_token,
            speech_tokens,
            self.convert_token,
            self.text_modality_token,
            text_tokens,
            self.stop_token
        ])
        return tokens

    def get_valid_sequence(self, split):
        methods = [
            self.get_tokens_speech,
            self.get_tokens_text,
            self.get_asr,
            self.get_tts,
            self.get_tokens_continue,
        ]

        while True:
            try:
                method = random.choice(methods)
                yield method(split)
            except EOFError as e:
                pass
            except FileNotFoundError as e:
                pass
            except Exception as e:
                print(e)

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

        for tokens in self.get_valid_sequence(split):
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


def train_omni(args):
    from datetime import datetime

    today = datetime.today().strftime('%y%m%d-%H%M%S')
    out_dir = Path(f'{CACHE_DIR}/models/omni_tasks_large/')

    interleaved_dir = [Path(f'{CACHE_DIR}/tinystories_omni/')]

    datasets_dirs = [
        Path(f'{CACHE_DIR}/jenny/'),
        Path(f'{CACHE_DIR}/expresso/'),
        Path(f'{CACHE_DIR}/mls_eng_10k/'),
        Path(f'{CACHE_DIR}/data/peoples_speech_tokens/'),
        Path(f'{CACHE_DIR}/data/gs_xl_en_tokens/'),
        Path(f'{CACHE_DIR}/hifi_tts'),
    ]

    data_generator = DataLoader(
        interleaved_dirs=interleaved_dir,
        datasets_dirs=datasets_dirs
    )

    pretrained = 'mdouglas/llmc-gpt2-774M-150B'
    vocab_size = cfg.VOCAB_SIZE

    model = GPT.from_pretrained(model_type='cmeraki/gpt2-124M-400B')
    model.expand_vocab(new_vocab_size=vocab_size)
    model.to(DEVICE)

    model = torch.compile(model)

    print(model)

    print("Vocab size: ", vocab_size)
    print("Model outdir: ", out_dir)
    print("Training omni".upper())


    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=training_cfg.STEPS,
              block_size=training_cfg.BLOCK_SIZE,
              eval_interval=training_cfg.EVAL_INTERVAL,
              batch_size=training_cfg.BATCH_SIZE,
              grad_accum_steps=training_cfg.GRAD_ACCUM_STEPS,
              device=DEVICE)

    return out_dir


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    train_omni(args)