import torch
import os
import tarfile

import pandas as pd
from glob import glob
import numpy as np

from gpt2_trainer import train as gpt_train
from gpt2_model import get_model, GPT
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files
from utils import Sample

from datalib import DataLoader, VOCAB_SIZES, OFFSET, PAD_TOKEN
from pathlib import Path
from utils import Sample

out_dir = Path('out_400b_ft')

DEVICE = 'cuda:0'

def get_vocab_size(source, target):
    vocab_size = max(OFFSET[source] + VOCAB_SIZES[source],
                    OFFSET[target] + VOCAB_SIZES[target],
                    PAD_TOKEN[source], PAD_TOKEN[target]) + 1

    return vocab_size


def get_flag(path):
    return (path / 'SUCCESS').exists()

def set_flag(path):
    with open(path / 'SUCCESS', 'w') as flag:
        flag.write('<COMPLETE>')


def get_dataset():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id='cmeraki/gsxl_tokens',
                             repo_type='dataset',
                             token=os.getenv('HF_TOKEN_CMERAKI'))

    path = Path(path)
    print("Dataset path:", path)

    if get_flag(path):
        print("Untar already completed, skipping")
        return path

    for tarname in glob(str(path / "*.tar")):
        print("Extracting", tarname)
        tf = tarfile.open(path / tarname)
        tf.extractall(path=path)
        tf.close()
        print("Deleting", tarname)
        os.remove(path / tarname)

    set_flag(path)

    return path


def iter_dataset(path):
    metadata_path = path / 'metadata.csv'
    df_metadata = pd.read_csv(metadata_path)

    columns = ['sid', 'text_tn']
    df_metadata = df_metadata[columns].to_dict(orient='records')

    idx = 0
    for example in df_metadata:
        
        idx += 1
        try:
            token_path = lambda x : path / x / f"{example['sid']}.npy"
            # example[SEMANTIC] = np.load(token_path(SEMANTIC))
            # example[ACOUSTIC] = np.load(token_path(ACOUSTIC))
            sample = Sample(text=example['text_tn'].lower(), id=example['sid'])
            yield sample
        except Exception as e:
            print("errors", e)


def decorate_instructions():
    # moved to instructlib.py
    pass


def prepare_data():
    path = get_dataset()
    return path
    # dataset = iter_dataset(path)
    # encode_files(dataset, path / TEXT, TEXT, device='cpu')
    

def train_translator(source, target, data_dir, tokenizer_config):
    print("===============")
    print(f"Training {source} {target}".upper())
    print("===============")

    vocab_size = get_vocab_size(source, target)
    print("Vocab size", vocab_size)

    model = GPT.from_pretrained('cmeraki/gpt2-124M-400B')
    model.expand_vocab(new_vocab_size=vocab_size)

    saved_model = torch.load("out_400b_ft/text_semantic/gpt_last.pt")['model']
    saved_model = {k.replace('_orig_mod.', ''): saved_model[k] for k in saved_model}

    model.load_state_dict(saved_model)
    
    model.to(DEVICE)

    model = torch.compile(model)

    print(model)
    
    data_generator = DataLoader(data_dir=data_dir,
                                source=source,
                                target=target, 
                                tokenizer_config)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=f'{out_dir}/{source}_{target}',
              steps=15000,
              block_size=1024,
              eval_interval=500,
              eval_steps=100,
              batch_size=32,
              grad_accum_steps=4,
              device=DEVICE)


def train():
    data_dir = prepare_data()
    # train_translator(TEXT, SEMANTIC, data_dir)
    train_translator(SEMANTIC, ACOUSTIC, data_dir)    
    # train_translator(SEMANTIC, TEXT, data_dir)


if __name__ == '__main__':
    train()