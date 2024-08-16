import glob
import torch
import random

import numpy as np
from pathlib import Path

from tts.gpt2_trainer import train as gpt_train
from tts.gpt2_model import get_model
from common import TEXT, SEMANTIC, ACOUSTIC, DEVICE
from common import Config as cfg
from tqdm import tqdm

print(cfg.__dict__)

class DataLoader:
    def __init__(self, 
                 data_dir, 
                 source, 
                 target, 
                 max_source_tokens=256, 
                 prompt_length=0, 
                 max_files=None):
        
        self.data_dir = data_dir
        self.source = source
        self.target = target
        self.max_files = max_files
        self.max_source_tokens = max_source_tokens
        self.prompt_length = prompt_length
        self.prompting = False
        
        self.files, self.filenames = self.load_files()
        
    def load_files(self):
        files = {}
        filenames = None


        for type in [self.source, self.target]:
            files[type] = {}
            file_iter = tqdm(glob.iglob(f"{self.data_dir}/{type}/*.npy"), desc=f'reading {type}')
            for f in file_iter:
                files[type][Path(f).name] = Path(f)             
                if self.max_files:
                    if len(files[type]) >= self.max_files:
                        break

            if not filenames:
                filenames = set(files[type].keys())

            filenames = filenames.intersection(set(files[type].keys()))

        filenames = list(filenames)

        print("Num files", len(filenames))

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
    

    @staticmethod
    def prepare_source(source_arr, source, max_source_tokens):
        source_arr = source_arr + cfg.OFFSET[source]
        source_arr = np.reshape(source_arr, -1)
        source_arr = source_arr[0: max_source_tokens]
        # bark trains with fixed size source 
        
        # source_arr = np.pad(
        #     source_arr,
        #         (0, max_source_tokens - len(source_arr)),
        #         constant_values=cfg.PAD_TOKEN[source],
        #         mode="constant",
        #     )
        return source_arr
    
    @staticmethod
    def prepare_target(target_arr, target):
        target_arr = target_arr + cfg.OFFSET[target]

        if target == ACOUSTIC:
            target_arr = target_arr[:cfg.coarse_codebooks]
            target_arr = DataLoader.codebook_encoding(target_arr, cfg.per_codebook_size)

        target_arr = target_arr.reshape(-1)

        return target_arr

    @staticmethod
    def prepare_prompt(prompt=None, prompt_length=0, target=ACOUSTIC):
        if prompt:
            prompt_arr = prompt[-prompt_length:]
            prompt_arr = np.pad(
            prompt_arr,
                (0, prompt_length - len(prompt_arr)),
                constant_values=cfg.PAD_TOKEN[target],
                mode="constant",
            )

        elif prompt_length > 0:
            prompt_arr = np.array([cfg.PAD_TOKEN[target]] * prompt_length)

        else:
            prompt_arr = []

        return prompt_arr

    def load_batch(self, split, block_size, batch_size):
        source = self.source
        target = self.target

        some_filenames = random.sample(self.filenames[split], batch_size)
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + cfg.STOP_TOKEN[target]
        y = y + cfg.STOP_TOKEN[target]

        for i in range(batch_size):
            f = some_filenames[i]
            source_arr = np.load(self.files[source][f])
            target_arr = np.load(self.files[target][f])

            source_arr = self.prepare_source(source_arr, 
                                             source=self.source, 
                                             max_source_tokens=self.max_source_tokens)
            
            target_arr = self.prepare_target(target_arr, target=self.target)
            
            prompt_arr = self.prepare_prompt(target_arr if self.prompting else None,
                                             target=self.target, 
                                             prompt_length=self.prompt_length)
            
            tokens = np.hstack([source_arr, prompt_arr, cfg.INFER_TOKEN[target], target_arr]).astype(np.int64)
            

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

def train_translator(source, target, data_dir, out_dir, pretrained=None, prompt_length=0):
    vocab_size = cfg.VOCAB_SIZE
    out_dir = out_dir / f'{source}_{target}'

    model = get_model(vocab_size=vocab_size,
                      device=DEVICE,
                      path=pretrained)

    print(f"{source}:{target} Vocab size", vocab_size)
    print("Model outdir", out_dir)
    print(f"Training {source} {target}".upper())

    data_generator = DataLoader(data_dir=data_dir,
                                source=source,
                                target=target,
                                prompt_length=prompt_length)


    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=16000,
              block_size=1024,
              eval_interval=100,
              eval_steps=10,
              batch_size=40,
              grad_accum_steps=8,
              device=DEVICE)

    return out_dir

def download_dataset(local_path):
    if Path(f'{local_path}/success').exists():
        print("Already downloaded")
        return

    from huggingface_hub import snapshot_download
    import tarfile

    snapshot_download('cmeraki/gs_xl_en_tokens', 
                      repo_type='dataset', 
                      local_dir=local_path)
    
    for tar_name in glob.glob(f"{local_path}/*.tar"):
        print(tar_name)
        tf = tarfile.open(tar_name)
        tf.extractall(path=local_path)
        tf.close()
    
    with open(f'{local_path}/success', 'w') as flag:
        flag.write('y')

def train():
    from common import cache_dir
    data_dir = f'{cache_dir}/data/gs_xl_en_tokens/'
    
    download_dataset(data_dir)
    out_dir = Path(f'{cache_dir}/data/models/mymodel/')
    
    train_translator(TEXT, SEMANTIC, data_dir, out_dir, prompt_length=0)
    train_translator(SEMANTIC, ACOUSTIC, data_dir, out_dir, prompt_length=0)
    train_translator(SEMANTIC, TEXT, data_dir, out_dir, prompt_length=0)
    
if __name__ == '__main__':
    train()