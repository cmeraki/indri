import os
import tarfile

import pandas as pd
from glob import glob
import numpy as np

from gpt2_trainer import train as gpt_train
from gpt2_model import get_model, GPT
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files
from utils import Sample

from datalib import DataLoader, VOCAB_SIZES, OFFSET
from pathlib import Path

out_dir = Path('out')

DEVICE = 'cpu'

def get_vocab_size(source, target):
    vocab_size = max(OFFSET[source] + VOCAB_SIZES[source], OFFSET[target] + VOCAB_SIZES[target])
    return vocab_size


def get_dataset():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id='cmeraki/gsxs_tokens',
                             repo_type='dataset',
                             token=os.getenv('HF_TOKEN_CMERAKI'))

    path = Path(path)
    for tarname in glob(str(path / "*.tar")):
        tf = tarfile.open(path / tarname)
        tf.extractall(path=path)

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
            for token_type in ['semantic', 'acoustic']:
                token_path = path / token_type / f"{example['sid']}.npy"
                if os.path.exists(token_path):
                    example[f'{token_type}_tokens'] = np.load(token_path)
            yield example
        except:
            pass


def decorate_instructions():
    # moved to instructlib.py
    pass


def prepare_data():
    path = get_dataset()
    for example in iter_dataset(path):
        # print(example)
        pass


def train_translator(source, target):
    print("===============")
    print(f"Training {source} {target}".upper())
    print("===============")

    vocab_size = get_vocab_size(source, target)
    print("Vocab size", vocab_size)

    model = GPT.from_pretrained('mdouglas/llmc-gpt2-124M-400B')
    model = model.expand_vocab(new_vocab_size=VOCAB_SIZES[TEXT] + VOCAB_SIZES[SEMANTIC] + 2)
    
    data_generator = DataLoader(data_dir=data_dir / dsname,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=f'{out_dir}/{source}_{target}',
              steps=16000,
              block_size=128,
              eval_interval=200,
              eval_steps=100,
              batch_size=2,
              grad_accum_steps=2,
              device=DEVICE)


def train():
    prepare_data()
    # train_translator(TEXT, TEXT)
    # train_translator(SEMANTIC, ACOUSTIC)


if __name__ == '__main__':
    train()