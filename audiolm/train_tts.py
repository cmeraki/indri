from gpt2_trainer import train as gpt_train
from gpt2_model import get_model
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files
from utils import iter_dataset

from datalib import DataLoader, VOCAB_SIZES, OFFSET
from pathlib import Path

dsname = 'speechcolab/gigaspeech'
data_dir = Path('../data')
out_dir = Path('out')

DEVICE = 'cuda:0'


def get_vocab_size(source, target):
    end = max(OFFSET[source] + VOCAB_SIZES[source], OFFSET[target] + VOCAB_SIZES[target])
    vocab_size = end + 3
    print(end, vocab_size)
    return vocab_size


def prepare_data():
    types = (SEMANTIC, ACOUSTIC, TEXT)
    for type in types:
        dataset = iter_dataset(repo=dsname,
                               name='s',
                               splits=['train'])

        encode_files(dataset=dataset,
                     outdir=data_dir/dsname/type,
                     type=type,
                     device=DEVICE)


def train_translator(source, target):
    print("===============")
    print(f"Training {source} {target}".upper())
    print("===============")
    
    vocab_size = get_vocab_size(source, target)
    print("Vocab size", vocab_size)

    model = get_model(n_layer=4,
                      n_head=4,
                      n_embd=256,
                      vocab_size=vocab_size,
                      block_size=1024,
                      compile=True,
                      device=DEVICE)

    data_generator = DataLoader(data_dir=data_dir/dsname,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=f'{out_dir}/{source}_{target}',
              steps=16000,
              block_size=1024,
              eval_interval=200,
              eval_steps=100,
              batch_size=16,
              grad_accum_steps=16,
              device=DEVICE)


def train():
    # prepare_data()
    # train_translator(TEXT, SEMANTIC)
    # train_translator(SEMANTIC, ACOUSTIC)
    train_translator(SEMANTIC, TEXT)


if __name__ == '__main__':
    train()