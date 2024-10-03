import json
import torch
import random
import numpy as np
from pathlib import Path

from tokenlib import get_tokenizer
from gpt2_trainer_with_ddp import train as gpt_train
from gpt2_model import GPT, get_model
from logger import get_logger

from commons import Config as cfg
from commons import CACHE_DIR, SPEAKER_FILE, TEXT, MIMI, CONVERT

logger = get_logger(__name__)
logger.info(cfg.__dict__)

def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]


def codebook_encoding(
    arr: torch.tensor,
    per_codebook_size: int):

    # Interleave n codebooks as 1
    c, n = arr.shape
    i_values = np.arange(c) * per_codebook_size
    arr += i_values.reshape(c, 1)
    flat_arr = arr.reshape(c * n, order='F')
    # print("-------------------------")
    # print(arr.shape, arr)
    # print(flat_arr.shape, flat_arr)
    # print("=======================")
    return flat_arr


def speaker_id_format(dataset, speaker):
    speaker_id = cfg.UNKNOWN_SPEAKER_ID
    if speaker:
        speaker_id = f'[spkr_{dataset}_{speaker}]'
    return speaker_id

def get_text_tokenizer():
    text_tokenizer = get_tokenizer(type=TEXT, device='cpu')

    for idx in range(cfg.VOCAB_SIZES[MIMI]):
        text_tokenizer.tokenizer.add_tokens(f'[aco_{idx}]')

    for tok in list(cfg.MODALITY_TOKENS.values()) + list(cfg.TASK_TOKENS.values()) + [cfg.STOP_TOKEN]:
        text_tokenizer.tokenizer.add_tokens(tok)
    
    # Create special tokens for each speaker
    for line in open(SPEAKER_FILE):
        sample = json.loads(line.strip())
        speaker_id = speaker_id_format(dataset=sample['dataset'],
                                       speaker=sample['speaker'])
        
        text_tokenizer.tokenizer.add_tokens(speaker_id)
        # logger.info(f'Added token {speaker_id}, {text_tokenizer.encode(speaker_id)}')

    text_tokenizer.tokenizer.add_tokens(cfg.UNKNOWN_SPEAKER_ID)
    return text_tokenizer

class DataLoader:
    def __init__(self, datasets_dirs, maxfiles=None):
        self.dataset_dirs = datasets_dirs
        self.load_parallel_data(self.dataset_dirs, maxfiles=maxfiles)
        self.text_tokenizer = get_text_tokenizer()
        self.bad_reads = {MIMI: 0}
        self.total_reads = {MIMI: 0}

    def load_parallel_data(self, dirs, maxfiles=None):
        metadata = {}
        for dir in dirs:
            print('loading dataset', dir)
            dir = Path(dir)
            metadata_path =  dir / 'annotation' / 'metadata.jsonl'
            for num_line, line in enumerate(open(metadata_path)):
                _metadata = json.loads(line.strip())
                _metadata['dir'] = dir
                _metadata['dataset'] = dir.name
                metadata[_metadata['id']] = _metadata
                if maxfiles and (num_line > maxfiles):
                    break
            
        logger.info(f"num metadata lines: {len(metadata)}")

        self.ids = list(metadata.keys())
        random.shuffle(self.ids)
        
        self.ids = {
            'train' : self.ids[1000:],
            'val' : self.ids[:1000],
        }

        self.metadata = metadata

    def normalize_text(self, text):
        text = text.lower()
        text = text.replace("<comma>", ',')
        text = text.replace("<period>", '.')
        text = text.replace('<questionmark>', '?')
        text = text.replace('<exclamationpoint>', '!')
        text = text.replace("\n", " ")
        return text

    def load_text_tokens(self, id):
        sample = self.metadata[id]
        text = sample['raw_text']
        norm_text = self.normalize_text(text)
        tokens = np.asarray(self.text_tokenizer.encode(norm_text)) + cfg.OFFSET[TEXT]
        return tokens

    def get_tokens_path(self, id, type):
        path = None
        sample = self.metadata[id]
        if sample[f'{type}_tokens'] is not None:
            path = sample['dir'] / sample[f'{type}_tokens'].replace('.opus', '').replace('.flac', '')
        return path

    def load_acoustic_tokens(self, id):
        self.total_reads[MIMI] += 1
        tokens = None
        
        path = str(self.get_tokens_path(id, MIMI)).replace(MIMI, 'mimi')
        tokens = np.load(path).astype(np.int64)
        tokens = tokens[:4]
        tokens = codebook_encoding(tokens, 2048)
        tokens = np.reshape(tokens, -1)
        tokens = tokens + cfg.OFFSET[MIMI]
    
        self.bad_reads[MIMI] += 1

        return tokens
    

    def load_speaker_id(self, id):
        null_speaker_id = self.text_tokenizer.encode(cfg.UNKNOWN_SPEAKER_ID)
        speaker_id = null_speaker_id
        if id:
            dataset = self.metadata[id]['dataset']
            ds_speaker_id = self.metadata[id]['speaker_id']
            text = speaker_id_format(dataset, ds_speaker_id)
            _speaker_id = self.text_tokenizer.encode(text)

            if len(_speaker_id) == 1:
                speaker_id = _speaker_id 

        return speaker_id
    

class TaskGenerator:
    def __init__(self, loader) -> None:
        self.text_tokenizer = get_text_tokenizer()
        self.loader = loader
        # make all task tokens
        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)

        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])
    
    def get_text_acoustic(self):
        tokens = np.hstack([
            self.text_modality_token,
            self.text_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self.speaker_id,
            self.acoustic_tokens,
            self.stop_token
        ])

        return tokens

    def set_data(self, split):
        id = random.choice(self.loader.ids[split])

        self.speaker_id = self.loader.load_speaker_id(id)
        self.acoustic_tokens = self.loader.load_acoustic_tokens(id)
        self.text_tokens = self.loader.load_text_tokens(id)


    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + self.stop_token
        y = y + self.stop_token

        # replacing leet code with unleet code
        batch_tasks = []
        for batch_idx in range(batch_size):
            while True:
                self.set_data(split=split)
                tokens = self.get_text_acoustic()
                task = 'mimi'
                # print(task)
                if tokens is not None:
                    break
                
            batch_tasks.append(task)
            _x = tokens[0:block_size]
            _y = tokens[1:block_size + 1]
            
            x[batch_idx][:len(_x)] = _x 
            y[batch_idx][:len(_y)] = _y

        return x, y, batch_tasks

    def get_batch(self, split, device, block_size, batch_size):
        x, y, tasks = self.load_batch(split,
                                block_size=block_size,
                                batch_size=batch_size)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if 'cuda' in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y, tasks

def train_text_semantic(dataset_dirs):
    # out_dir = Path(f'{CACHE_DIR}/models/mimi_gigaspeech/')

    # data_generator = DataLoader(
    #     datasets_dirs=dataset_dirs
    # )

    # taskgen = TaskGenerator(loader=data_generator)

    # from tqdm import tqdm
    # for i in tqdm(range(100000)):
    #     batch = taskgen.get_batch(block_size=1024, batch_size=1, device='cpu', split='train')
    #     print(batch)
        # print(data_generator.bad_reads, data_generator.total_reads)

    pretrained = 'mdouglas/llmc-gpt2-774M-150B'
    # pretrained = 'cmeraki/gpt2-124M-400B'
    vocab_size = cfg.VOCAB_SIZE

    model = GPT.from_pretrained(model_type=pretrained)
    model.expand_vocab(new_vocab_size=vocab_size)

    state_dict = torch.load('/home/apurva/Downloads/gpt_46000.pt')
    print(state_dict)
    
    # logger.info(model)    

    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Model outdir: {out_dir}")
    logger.info("Training text sem".upper())

    # gpt_train(model,
    #           get_batch=taskgen.get_batch,
    #           out_dir=out_dir,
    #           steps=10000,
    #           block_size=1024,
    #           eval_interval=1000,
    #           batch_size=2,
    #           grad_accum_steps=2)



if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    
    args = parser.parse_args()
    
    reading_datasets = ['gigaspeech']
    
    speaking_datasets = []
    
    datasets = reading_datasets
    print("Datasets=", datasets)
    dirs = [Path(f'{CACHE_DIR}/{dsname}/') for dsname in datasets]
    train_text_semantic(dataset_dirs=dirs)