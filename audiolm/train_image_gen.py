import glob

from gpt2_trainer import train as gpt_train
from gpt2_model import get_model, GPT
from tokenlib import SEMANTIC, TEXT, IMAGE, encode_files
from utils import iter_dataset

from datalib import DataLoader, VOCAB_SIZES, OFFSET, PAD_TOKEN
from pathlib import Path
import zipfile
import io
from tqdm import tqdm
from utils import Sample
import torch

dsname = 'laion_coco'
image_dataset_dir = '/media/apurva/data/data/images/laion-coco-aesthetic/'
data_dir = Path('../data/tokens/')
out_dir = Path('out')

DEVICE = 'cuda:0'

def get_vocab_size(source, target):
    vocab_size = max([OFFSET[source] + VOCAB_SIZES[source],
                      OFFSET[target] + VOCAB_SIZES[target],
                      PAD_TOKEN[source], 
                      PAD_TOKEN[target]]) + 1
    return vocab_size 


def stream_from_zip(zip_file_path, extension):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            if file_name.endswith(extension):
                with zip_ref.open(file_name) as file:
                    content = file.read()
                    bytes_io = io.BytesIO(content)
                    yield file_name, bytes_io


def iter_dataset(type):
    import json

    annotations = '/media/apurva/data/data/images/laion-coco-aesthetic/annotations.jsonl'
    image_dataset_dir = '/media/apurva/data/data/images/laion-coco-aesthetic/images/*.zip'

    annotations_dict = {}

    for x in tqdm(open(annotations), desc='reading annotations'):
        js = json.loads(x)
        id = js['id']
        caption = js['caption']
        annotations_dict[id] = caption

    # last iteration at 1001702

    if type == IMAGE:
        zipfiles = list(sorted(list(glob.glob(image_dataset_dir))))
        print("Num files:::", len(zipfiles))
        for zf in sorted(zipfiles):
            for fname, image_path in stream_from_zip(zf, extension='.jpg'):
                id = fname.split('/')[-1].split('.')[0]
                if id not in annotations_dict:
                    continue

                example = Sample(audio_path=None,
                                 text=annotations_dict[id].lower(),
                                 id=id,
                                 image_path=image_path)
                yield example

    if type == TEXT:
        for id in annotations_dict:
            example = Sample(audio_path=None,
                             text=annotations_dict[id].lower(),
                             id=id,
                             image_path=None)
            yield example

def prepare_data():
    types = [IMAGE]
    for type in types:
        dataset = iter_dataset(type)

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
    
    # model = GPT.from_pretrained('mdouglas/llmc-gpt2-124M-400B')
    
    # model.expand_vocab(vocab_size)
    
    # model = model.to(DEVICE)
    # model = torch.compile(model)
    
    model = get_model(n_layer=8,
                      n_head=8,
                      n_embd=512,
                      vocab_size=vocab_size,
                      block_size=1024,
                      compile=True,
                      device=DEVICE)

    data_generator = DataLoader(data_dir=data_dir / dsname,
                                source=source,
                                target=target)

    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=f'{out_dir}/{source}_{target}',
              steps=600000,
              block_size=1024,
              eval_interval=10000,
              eval_steps=100,
              batch_size=16,
              grad_accum_steps=2,
              device=DEVICE)


def train():
    # prepare_data()
    # train_translator(TEXT, IMAGE)
    train_translator(IMAGE, IMAGE)
    # train_translator(SEMANTIC, ACOUSTIC)


if __name__ == '__main__':
    train()