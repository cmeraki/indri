from gpt2_trainer import train as gpt_train
from gpt2_model import get_model
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files
from utils import Sample

from datalib import DataLoader, VOCAB_SIZES, OFFSET
from pathlib import Path

from datasets import load_dataset

dsname = 'fineweb_xxs'
data_dir = Path('../data')
out_dir = Path('out')

DEVICE = 'cpu'


def get_vocab_size(source, target):
    end = max(OFFSET[source] + VOCAB_SIZES[source], OFFSET[target] + VOCAB_SIZES[target])
    vocab_size = end + 3
    return vocab_size


def iter_dataset():
    fw = load_dataset("HuggingFaceFW/fineweb",
                      name="sample-10BT",
                      split="train",
                      streaming=True)
    id = 0
    for elem in fw:
        id += 1
        sample = Sample(id=f"{id}", text=elem['text'])
        yield sample


def prepare_data():
    dataset = iter_dataset()

    encode_files(dataset=dataset,
                 outdir=data_dir / dsname / TEXT,
                 type=TEXT,
                 device=DEVICE)


def train_translator(source, target):
    print("===============")
    print(f"Training {source} {target}".upper())
    print("===============")

    vocab_size = get_vocab_size(source, target)
    print("Vocab size", vocab_size)

    model = get_model(n_layer=2,
                      n_head=2,
                      n_embd=64,
                      vocab_size=vocab_size,
                      block_size=256,
                      compile=False,
                      device=DEVICE)

    data_generator = DataLoader(data_dir=data_dir / dsname,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=f'{out_dir}/{source}_{target}',
              steps=16000,
              block_size=256,
              eval_interval=200,
              eval_steps=100,
              batch_size=16,
              grad_accum_steps=16,
              device=DEVICE)


def train():
    prepare_data()
    train_translator(TEXT, TEXT)


if __name__ == '__main__':
    train()