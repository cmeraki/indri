import pandas as pd

from gpt2_trainer import train as gpt_train
from gpt2_model import get_model, GPT
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files
from utils import Sample

from datalib import DataLoader, VOCAB_SIZES, OFFSET
from pathlib import Path

dsname = 'isotonic_instruct'
data_dir = Path('../data')
out_dir = Path('out')

DEVICE = 'cpu'


def get_vocab_size(source, target):
    end = max(OFFSET[source] + VOCAB_SIZES[source], OFFSET[target] + VOCAB_SIZES[target])
    vocab_size = end + 3
    print(end, vocab_size)
    return vocab_size

def iter_dataset():
    from datasets import load_dataset
    # gs = load_dataset('Isotonic/human_assistant_conversation', token='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')
    # splits = ['test']

    dataset = pd.read_parquet('test.parquet')
    print(dataset)

    idx = 0
    for example in dataset['text']:
        idx += 1
        example = Sample(audio_path="",
                         text=example,
                         id=f"id_{idx}")
        yield example


def prepare_data():
    types = [TEXT]
    for type in types:
        dataset = iter_dataset()

        encode_files(dataset=dataset,
                     outdir=data_dir / dsname / type,
                     type=type,
                     device=DEVICE)


def train_translator(source, target):
    print("===============")
    print(f"Training {source} {target}".upper())
    print("===============")

    vocab_size = get_vocab_size(source, target)
    print("Vocab size", vocab_size)

    model = GPT.from_pretrained('mdouglas/llmc-gpt2-124M-400B')

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
    train_translator(TEXT, TEXT)
    # train_translator(SEMANTIC, ACOUSTIC)


if __name__ == '__main__':
    train()