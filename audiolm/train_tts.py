from train import train as gpt_train
from train import get_model
from tokenlib import SEMANTIC, ACOUSTIC, TEXT, encode_files

from datalib import DataLoader
from pathlib import Path

data_dir = Path('../data')
out_dir = Path('out')


def prepare_data():
    encode_files(dataset='speechcolab/gigaspeech', outdir=outdir, type=type)


def train_text_semantic():
    text_semantic_model = get_model(n_layer=4,
                                    n_head=4,
                                    n_embd=256,
                                    vocab_size=50000,
                                    block_size=1024)


    data_generator = DataLoader(data_dir=data_dir/'audio_tokens/gigaspeech_xs',
                               source=TEXT,
                               target=SEMANTIC)

    gpt_train(text_semantic_model,
          get_batch=data_generator.get_batch,
          out_dir=f'{out_dir}/text_semantic',
          steps=3000,
          block_size=1024,
          eval_interval=200,
          eval_steps=100,
          batch_size=32)


def train_semantic_acoustic():
    semantic_acoustic_model = get_model(n_layer=4,
                      n_head=4,
                      n_embd=256,
                      vocab_size=3072,
                      block_size=1024)

    data_generator = semantic_acoustic_generator(data_dir)
    gpt_train(semantic_acoustic_model,
          get_batch=data_generator,
          out_dir=f'{out_dir}/semantic_acoustic',
          steps=3000,
          block_size=1024,
          eval_interval=5,
          eval_steps=4,
          batch_size=64)


def train():
    # prepare_data()
    train_text_semantic()
    # train_semantic_acoustic()

train()