import json
import torch
import random
import numpy as np
from pathlib import Path

from datalib.tokenlib import get_tokenizer
from omni.gpt2_trainer import train as gpt_train
from omni.gpt2_model import GPT, get_model
from omni.logger import get_logger

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR, SPEAKER_FILE
from configs.constants import *
import configs.training_omni as training_cfg

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
    return flat_arr


def speaker_id_format(dataset, speaker):
    speaker_id = cfg.UNKNOWN_SPEAKER_ID
    if speaker:
        speaker_id = f'[spkr_{dataset}_{speaker}]'
    return speaker_id

def get_text_tokenizer():
    text_tokenizer = get_tokenizer(type=TEXT, device='cpu')
    for idx in range(cfg.VOCAB_SIZES[SEMANTIC]):
        text_tokenizer.tokenizer.add_tokens(f'[sem_{idx}]')

    for idx in range(cfg.VOCAB_SIZES[ACOUSTIC]):
        text_tokenizer.tokenizer.add_tokens(f'[aco_{idx}]')

    for tok in list(cfg.MODALITY_TOKENS.values()) + list(cfg.TASK_TOKENS.values()) + [cfg.STOP_TOKEN]:
        text_tokenizer.tokenizer.add_tokens(tok)
    
    # Create special tokens for each speaker
    for line in open(SPEAKER_FILE):
        sample = json.loads(line.strip())
        speaker_id = speaker_id_format(dataset=sample['dataset'],
                                       speaker=sample['speaker'])
        
        text_tokenizer.tokenizer.add_tokens(speaker_id)
        logger.info(f'Added token {speaker_id}, {text_tokenizer.encode(speaker_id)}')

    text_tokenizer.tokenizer.add_tokens(cfg.UNKNOWN_SPEAKER_ID)
    return text_tokenizer

class DataLoader:
    def __init__(self, datasets_dirs, maxfiles=None):
        self.dataset_dirs = datasets_dirs
        self.load_parallel_data(self.dataset_dirs, maxfiles=maxfiles)
        self.text_tokenizer = get_text_tokenizer()
        self.bad_reads = {SEMANTIC: 0 , ACOUSTIC: 0}
        self.total_reads = {SEMANTIC: 0, ACOUSTIC: 0}

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

    def load_semantic_tokens(self, id):
        self.total_reads[SEMANTIC] += 1
        tokens = None
        path = self.get_tokens_path(id, SEMANTIC)
        try:
            tokens = np.load(path).reshape(-1) + cfg.OFFSET[SEMANTIC]
            tokens = tokens.reshape(-1)
            tokens = replace_consecutive(tokens)
        except:
            self.bad_reads[SEMANTIC] += 1
            # print(path, id, self.metadata[id])
            pass    
        return tokens

    def load_acoustic_tokens(self, id):
        self.total_reads[ACOUSTIC] += 1
        tokens = None
        try:
            tokens = np.load(self.get_tokens_path(id, ACOUSTIC)).astype(np.int64)
            tokens = tokens[:cfg.coarse_codebooks]
            tokens = codebook_encoding(tokens, cfg.per_codebook_size)
            tokens = np.reshape(tokens, -1)
            tokens = tokens + cfg.OFFSET[ACOUSTIC]
        except:
            self.bad_reads[ACOUSTIC] += 1
            pass

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
    def __init__(self, loader, full_batches) -> None:
        self.text_tokenizer = get_text_tokenizer()
        self.loader = loader
        # make all task tokens
        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)

        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.semantic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[SEMANTIC])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[ACOUSTIC])
        self.full_batches = full_batches
    
    def get_text_completion(self):
        tokens = np.hstack([
            self.text_modality_token,
            self.text_tokens,
            self.stop_token
        ])
        return tokens

    def get_semantic_completion(self):
        tokens = np.hstack([
            self.semantic_modality_token,
            self.speaker_id,
            self.semantic_tokens,
            self.stop_token
        ])
        return tokens

    def get_text_semantic(self):
        tokens = np.hstack([
            self.text_modality_token,
            self.text_tokens,
            self.convert_token,
            self.semantic_modality_token,
            self.speaker_id,
            self.semantic_tokens,
            self.stop_token
        ])

        return tokens
    
    def get_semantic_text(self):
        tokens = np.hstack([
            self.semantic_modality_token,
            self.speaker_id,
            self.semantic_tokens,
            self.convert_token,
            self.text_modality_token,
            self.text_tokens,
            self.stop_token
        ])
        return tokens

    def get_acoustic_completion(self):
        return np.hstack([
            self.acoustic_modality_token,
            self.speaker_id,
            self.acoustic_tokens,
            self.stop_token
        ])

    def get_semantic_completion_without_speaker_id(self):
        return np.hstack([
            self.semantic_modality_token,
            self.semantic_tokens,
            self.stop_token
        ])

    def get_semantic_acoustic(self):
        return np.hstack([
            self.semantic_modality_token,
            self.semantic_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self.speaker_id,
            self.acoustic_tokens,
            self.stop_token
        ])

    def get_acoustic_semantic(self):
        return np.hstack([
            self.acoustic_modality_token,
            self.speaker_id,
            self.acoustic_tokens,
            self.convert_token,
            self.semantic_modality_token,
            self.semantic_tokens,
            self.stop_token
        ])
    
    def set_data(self, split):
        id = random.choice(self.loader.ids[split])

        self.speaker_id = self.loader.load_speaker_id(id)
        self.semantic_tokens = self.loader.load_semantic_tokens(id)
        self.acoustic_tokens = self.loader.load_acoustic_tokens(id)
        self.text_tokens = self.loader.load_text_tokens(id)

    def get_sample_text_semantic(self):
        if self.semantic_tokens is not None:
            methods = [self.get_semantic_completion]

        if self.text_tokens is not None:
            methods = [self.get_text_completion]

        if (self.text_tokens is not None) and (self.semantic_tokens is not None):
                methods = [self.get_text_semantic,
                    self.get_semantic_text, 
                    self.get_text_completion, 
                    self.get_semantic_completion]

        result = None
        task = None

        if methods:
            method = random.choice(methods)
            result = method()
            task = method.__name__
        return result, task


    def get_sample_semantic_acoustic(self):
        methods = []
        # print(self.semantic_tokens, self.acoustic_tokens)
        if self.semantic_tokens is not None:
            methods = [self.get_semantic_completion_without_speaker_id]

        if self.acoustic_tokens is not None:
            methods = [self.get_acoustic_completion]

        if (self.acoustic_tokens is not None) and (self.semantic_tokens is not None):
            methods = [self.get_semantic_acoustic,
                        # self.get_semantic_completion_without_speaker_id, 
                        # self.get_acoustic_semantic,
                        # self.get_acoustic_completion
                      ]

        result = None
        task = None

        if methods:
            method = random.choice(methods)
            result = method()
            task = method.__name__
        return result, task

    def set_generator(self, generator):
        self.sample_generator = generator

    def load_batch_full(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + self.stop_token
        y = y + self.stop_token

        # replacing leet code with unleet code
        batch_tasks = []
        for batch_idx in range(batch_size):
            sample_tokens = []
            sample_tasks = []
            length_now = 0

            while True:
                self.set_data(split=split)
                tokens, task = self.sample_generator()
                # print(task)
                if tokens is None:
                    continue
                
                sample_tasks.append(task)
                sample_tokens.append(tokens)

                length_now += len(tokens)
                if length_now > block_size + 1:
                    break
            
            
            batch_tasks.append(sample_tasks)
            sample_tokens = np.hstack(sample_tokens)

            x[batch_idx] = sample_tokens[0:block_size]
            y[batch_idx] = sample_tokens[1:block_size + 1]

        return x, y, batch_tasks

    
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
            sample_tokens = []
            sample_tasks = []
            length_now = 0

            while True:
                self.set_data(split=split)
                tokens, task = self.sample_generator()
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
        if self.full_batches:
            x, y, tasks = self.load_batch_full(split,
                                   block_size=block_size,
                                   batch_size=batch_size)

        else:
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

def train_text_semantic(device, dataset_dirs):
    out_dir = Path(f'{CACHE_DIR}/models/omni_text_sem_good_readin_medium/')

    data_generator = DataLoader(
        datasets_dirs=dataset_dirs
    )

    taskgen = TaskGenerator(loader=data_generator, full_batches=True)
    taskgen.set_generator(taskgen.get_sample_text_semantic)

    # from tqdm import tqdm
    # for i in tqdm(range(100000)):
    #     taskgen.get_batch(block_size=1024, batch_size=32, device='cpu', split='train')
        # print(data_generator.bad_reads, data_generator.total_reads)

    pretrained = 'gpt2-medium'
    vocab_size = cfg.VOCAB_SIZE

    # model = GPT.from_pretrained(model_type=pretrained)
    # model.expand_vocab(new_vocab_size=vocab_size)
    # model.to(device)


    model = get_model(
        model_type='gpt2-medium',
        vocab_size=vocab_size,
        block_size=1024,
        bias=True,
        device=device,
        path='/home/.cache/indri/models/omni_text_sem_good_readin_medium/gpt_last.pt'
    )

    
    model = torch.compile(model)

    logger.info(model)

    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Model outdir: {out_dir}")
    logger.info("Training text sem".upper())

    gpt_train(model,
              get_batch=taskgen.get_batch,
              out_dir=out_dir,
              steps=100000,
              block_size=1024,
              eval_interval=1000,
              batch_size=16,
              grad_accum_steps=4,
              device=device)


def train_semantic_acoustic(device, dataset_dirs):
    out_dir = Path(f'{CACHE_DIR}/models/omni_sem_aco_911_mukesh/')

    data_generator = DataLoader(
        datasets_dirs=dataset_dirs
    )

    taskgen = TaskGenerator(loader=data_generator, full_batches=False)
    taskgen.set_generator(taskgen.get_sample_semantic_acoustic)

    # for i in range(100000):
    #     taskgen.get_batch(block_size=3072, batch_size=8, device='cpu', split='train')
    #     print(data_generator.bad_reads, data_generator.total_reads)

    vocab_size = cfg.VOCAB_SIZE

    model = get_model(
        model_type='gpt2-medium',
        vocab_size=vocab_size,
        block_size=3072,
        bias=True,
        device=device,
        path='/home/.cache/indri/models/omni_sem_aco_911_mukesh/gpt_last.pt'
    )

    model = torch.compile(model)

    logger.info(model)

    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Model outdir: {out_dir}")
    logger.info("Training sem aco".upper())

    gpt_train(model,
              get_batch=taskgen.get_batch,
              out_dir=out_dir,
              steps=70000,
              block_size=3072,
              eval_interval=1000,
              batch_size=8,
              grad_accum_steps=2,
              device=device, 
              start_step=60000)

def read_files_once(device, dataset_dirs):

    loader = DataLoader(
        datasets_dirs=dataset_dirs,
        maxfiles=None,
    )

    from tqdm import tqdm
    for id in tqdm(loader.ids['train']):
        # speaker_id = loader.load_speaker_id(id)
        # semantic_tokens = loader.load_semantic_tokens(id)
        # acoustic_tokens = loader.load_acoustic_tokens(id)
        try:
            text_tokens = loader.load_text_tokens(id)
        except Exception as e:
            print(id, e)

    # for i in range(100000):
    #     taskgen.get_batch(block_size=3072, batch_size=8, device='cpu', split='train')
    #     print(data_generator.bad_reads, data_generator.total_reads)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--type', type=str, required=True, help='one of textsem/semaco')
    
    args = parser.parse_args()
    
    reading_datasets = ['jenny',
                     'expresso',
                     'mls_eng_10k',
                     'gs_xl_en_tokens',
                     'hifi_tts',
                    ]
    
    speaking_datasets = ['youtube_en_asmr', 
                     'peoples_speech_tokens', 
                     'audiobooks_attenborough',]
    
    if args.type == 'textsem':
        datasets = reading_datasets
        print("Datasets=", datasets)
        dirs = [Path(f'{CACHE_DIR}/{dsname}/') for dsname in datasets]
        train_text_semantic(args.device, dataset_dirs=dirs)

    if args.type == 'semaco':
        datasets = reading_datasets + speaking_datasets
        print("Datasets=", datasets)
        dirs = [Path(f'{CACHE_DIR}/{dsname}/') for dsname in datasets]
        train_semantic_acoustic(args.device, dataset_dirs=dirs)

    if args.type == 'read':
        datasets = reading_datasets + speaking_datasets
        print("Datasets=", datasets)
        dirs = [Path(f'{CACHE_DIR}/{dsname}/') for dsname in datasets]
        read_files_once(args.device, dataset_dirs=dirs)
    