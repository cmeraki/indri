import torch

from gpt2_trainer import train as gpt_train
from gpt2_model import GPT
from tokenlib import SEMANTIC, ACOUSTIC, TEXT
from utils import Sample

from datalib import DataLoader, VOCAB_SIZES, OFFSET, PAD_TOKEN
from pathlib import Path

DEVICE = 'cuda:0'

def get_vocab_size(source, target):
    vocab_size = max(OFFSET[source] + VOCAB_SIZES[source],
                    OFFSET[target] + VOCAB_SIZES[target],
                    PAD_TOKEN[source], PAD_TOKEN[target]) + 1

    return vocab_size


def load_model(expanded_vocab_size=None, weights_path=None):
    model = GPT.from_pretrained('cmeraki/gpt2-124M-400B')

    if expanded_vocab_size:
        model.expand_vocab(new_vocab_size=expanded_vocab_size)

    if weights_path:
        saved_model = torch.load(weights_path)['model']
        saved_model = {k.replace('_orig_mod.', ''): saved_model[k] for k in saved_model}

        model.load_state_dict(saved_model)

    model.to(DEVICE)

    model = torch.compile(model)

    return model


def train_translator(source, target, data_dir, model, out_dir):
    print(f"Training {source} {target}".upper())

    data_generator = DataLoader(data_dir=data_dir,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=15000,
              block_size=1024,
              eval_interval=500,
              eval_steps=100,
              batch_size=32,
              grad_accum_steps=4,
              device=DEVICE)

    return out_dir

def train():

    out_dir = Path('out_400b_ft')
    data_dir = ''

    source, target = TEXT, SEMANTIC
    vocab_size = get_vocab_size(source, target)
    print(f"{source}:{target} Vocab size", vocab_size)

    text_semantic_model = load_model(vocab_size)
    text_semantic_model_path = train_translator(TEXT,
                                                SEMANTIC,
                                                model=text_semantic_model,
                                                data_dir=data_dir,
                                                out_dir=out_dir)

    source, target = SEMANTIC, ACOUSTIC
    vocab_size = get_vocab_size(source, target)
    print(f"{source}:{target} Vocab size", vocab_size)

    text_semantic_model = load_model(vocab_size)
    text_semantic_model_path = train_translator(TEXT,
                                                SEMANTIC,
                                                model=text_semantic_model,
                                                data_dir=data_dir,
                                                out_dir=out_dir)


if __name__ == '__main__':
    train()