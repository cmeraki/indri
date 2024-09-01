import os
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from tts.config import Config as cfg
from tts.config import TEXT, SEMANTIC, ACOUSTIC, DEVICE

from tts.gpt2_trainer import train as gpt_train
from tts.gpt2_model import get_model

print(cfg.__dict__)

class DataLoader:
    def __init__(
        self,
        data_dir,
        source,
        target,
        max_source_tokens=256,
        prompt_length=0,
        max_files=None
    ):

        self.data_dir = data_dir
        self.source = source
        self.target = target
        self.max_files = max_files
        self.max_source_tokens = max_source_tokens
        self.prompt_length = prompt_length
        self.prompting = False

        self.files, self.filenames = self.load_files()

        print(f'Training {source} to {target} with max source tokens {max_source_tokens} and max files {max_files}')

    def load_files(self):
        files = {}
        filenames = None

        for type in [self.source, self.target]:
            files[type] = {}
            file_iter = tqdm(glob.iglob(f"{self.data_dir}/{type}/**/*.npy", recursive=True), desc=f'reading {type}')
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
    def codebook_encoding(
        arr: torch.tensor,
        per_codebook_size: int
    ):

        # Interleave n codebooks as 1
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

        return source_arr

    @staticmethod
    def prepare_target(target_arr, target):
        target_arr = target_arr + cfg.OFFSET[target]

        if target == ACOUSTIC:
            target_arr = target_arr[:cfg.coarse_codebooks]
            target_arr = DataLoader.codebook_encoding(target_arr, cfg.per_codebook_size)

        target_arr = target_arr.reshape(-1)

        return target_arr

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
            
            tokens = np.hstack([source_arr, cfg.INFER_TOKEN[target], target_arr]).astype(np.int64)
            
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

def train_translator(source, target, data_dir, out_dir, pretrained=None, prompt_length=0):
    out_dir = os.path.join(out_dir, f'{source}_{target}')

    model = get_model(
        model_type=cfg.MODEL_TYPE,
        vocab_size=cfg.VOCAB_SIZE,
        block_size=cfg.BLOCK_SIZE[source],
        device=DEVICE,
        path=pretrained
    )

    print(f"Training {cfg.MODEL_TYPE} {source} {target}".upper())
    print(f"{source}:{target} Vocab size", cfg.VOCAB_SIZE)
    print("Model outdir", out_dir)

    data_generator = DataLoader(
        data_dir=data_dir,
        source=source,
        target=target,
        max_source_tokens=cfg.MAX_SOURCE_TOKENS[source],
        prompt_length=prompt_length
    )

    gpt_train(
        model,
        get_batch=data_generator.get_batch,
        out_dir=out_dir,
        steps=cfg.STEPS,
        block_size=cfg.BLOCK_SIZE[source],
        eval_interval=cfg.EVAL_INTERVAL,
        eval_steps=cfg.EVAL_STEPS,
        batch_size=cfg.BATCH_SIZE,
        grad_accum_steps=cfg.GRAD_ACCUM_STEPS,
        device=DEVICE
    )

    return out_dir

def download_dataset(local_path):
    print(f'Downloading dataset at {local_path}')

    import tarfile
    from huggingface_hub import snapshot_download

    datasets = [
        # 'cmeraki/expresso',
        'cmeraki/gs_xl_en_tokens',
        # 'cmeraki/wavcaps',
        # 'cmeraki/jenny',
        # 'cmeraki/peoples_speech_tokens'
    ]
    for dataset_name in datasets:
        if Path(os.path.join(local_path, dataset_name.split('/')[-1], 'success')).exists():
            print(f"Already downloaded {dataset_name}")
            continue

        print(f"Downloading data at {local_path}")
        snapshot_download(
            dataset_name,
            repo_type='dataset',
            local_dir=os.path.join(local_path)
        )

    # for tar_name in glob.glob(f"{local_path}/**/*.tar"):
    #     print(tar_name)
    #     tf = tarfile.open(tar_name)
    #     tf.extractall(path=os.path.join(local_path))
    #     tf.close()

    # with open(f'{local_path}/success', 'w') as flag:
    #     flag.write('y')

def train():
    from common import cache_dir

    data_dir = Path(os.path.join(cache_dir, 'romit', 'data'))
    out_dir = Path(os.path.join(cache_dir, 'romit', 'models', 'medium'))

    print("DATA DIR: ", data_dir)
    print("OUT DIR: ", out_dir)

    # download_dataset(data_dir)

    # train_translator(TEXT, SEMANTIC, data_dir, out_dir, prompt_length=0)
    train_translator(SEMANTIC, ACOUSTIC, data_dir, out_dir, prompt_length=0)
    # train_translator(SEMANTIC, TEXT, data_dir, out_dir, prompt_length=0)

if __name__ == '__main__':
    train()
