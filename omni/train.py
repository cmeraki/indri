import glob
import torch
import random

import numpy as np
from pathlib import Path

from tts.gpt2_trainer import train as gpt_train
from tts.gpt2_model import get_model, GPT
from common import DEVICE
from common import Config as cfg

print(cfg.__dict__)

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = self.load_files(self.data_dir)

    def load_files(self, data_dir):
        files = glob.glob(f'{data_dir}/*.npy')
        
        files = list(files)

        print("Num files", len(files))

        files = {
            'train': files[1000:],
            'val': files[:1000]
        }

        return files

    def load_batch(self, split, block_size, batch_size):
        some_filenames = random.sample(self.files[split], batch_size)

        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + cfg.OMNI_STOP_TOKEN
        y = y + cfg.OMNI_STOP_TOKEN

        for i in range(batch_size):
            tokens = np.load(some_filenames[i])
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


def train_omni(data_dir, out_dir, pretrained=None):
    vocab_size = cfg.VOCAB_SIZE
    out_dir = out_dir / 'omni'

    # model = get_model(vocab_size=vocab_size,
    #                   device=DEVICE,
    #                   path=pretrained)

    model = GPT.from_pretrained(pretrained)
    model.expand_vocab(new_vocab_size=vocab_size)
    model.to(DEVICE)

    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

    print(model)

    print("Vocab size", vocab_size)
    print("Model outdir", out_dir)
    print("Training omni".upper())
    
    data_generator = DataLoader(data_dir=data_dir)


    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=1000,
              block_size=1024,
              eval_interval=100,
              batch_size=32,
              grad_accum_steps=2,
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
    data_dir = f'instruct_tokens/'
    
    # download_dataset(data_dir)
    out_dir = Path(f'{cache_dir}/data/models/omni/')
    
    train_omni(data_dir, out_dir, pretrained='cmeraki/gpt2-124M-400B')

    # dl = DataLoader(data_dir)
    # batch = dl.get_batch('train', DEVICE, 1024, 1)
    # print(batch)

if __name__ == '__main__':
    train()