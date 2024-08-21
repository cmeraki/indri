import json
import tarfile
import io
import torchaudio
import torch

import numpy as np
from torio.io import CodecConfig
from tqdm import tqdm
from pathlib import Path

from tts.utils import convert_audio
from datalib.datalib import Dataset
from datalib.tokenlib import get_tokenizer, ACOUSTIC, SEMANTIC, TEXT
from datasets import load_dataset



def stream_samples():
    #{'audio': {'path': '8862_10456_000051.opus', 'array': array([ 0.00041701,  0.00012113,..]), 
    # 'sampling_rate': 16000}, 
    # 'original_path': 'http://www.archive.org/download/tobit_kjv_1512_librivox/tobit_03_kjv_64kb.mp3', 
    # 'begin_time': 83.25, 'end_time': 93.42, 
    # 'transcript': 'command therefore that i may now be delivered out of this distress and go into the everlasting place turn not thy face away from me', 'audio_duration': 10.170000000000002, 'speaker_id': '8862', 'book_id': '10456'}


    # ds = load_dataset(path="parler-tts/mls_eng_10k")
    import pandas as pd
    import pyarrow as pa

    # Read the .arrow file'
    df = pd.read_parquet('/home/apurva/Downloads/00e91c703ef82e0a78988fa32dd206f3e0c048047598bfaec3385f613b1dc6fe.parquet')

    for index, elem in df.iterrows():
        item = {
            "id": elem['audio']['path'],
            "audio": elem['audio']['bytes'],
            "sample_rate": 16000,
            "text": elem['transcript'],
            "duration_ms": elem['end_time'] - elem['end_time'],
            "speaker": elem['speaker_id']
        }
        yield item

@torch.inference_mode()
def make_dataset():
    dataset = Dataset(repo_id='mls_eng_10k', audio_format='')
    for item in tqdm(stream_samples()):
        id = item['id']
        if dataset.has(id):
            continue
        
        sample = dataset.create_sample(id=id)
        sample.raw_text = item['text']
        sample.speaker_id = item['speaker']
        sample.sample_rate = item['sample_rate']
        
        path = dataset.get_absolute_path(sample.audio_path)
        
        with open(path, 'wb') as f:
            audio_array = item['audio']
            f.write(audio_array)
        
        dataset.add_sample(sample)

    
    # dataset.upload(hf_repo_id='mls_eng_10k')


def tokenize():
    import audiotoken
    import os
    from datalib.tokenlib import AUDIO
    
    dataset = Dataset(repo_id='mls_eng_10k')
    from glob import glob
    path = str(dataset.dirs[AUDIO] / "*.opus")
    print(path)
    files = glob(path)
    bad = 0

    print("bad", bad)
    print("nfiles", len(files))
    print("from", dataset.dirs[AUDIO], "to", dataset.dirs[SEMANTIC])

    tokenizer = audiotoken.AudioToken(tokenizer='semantic_s', device='cuda:0')
    tokenizer.encode_batch_files(audio_files=files,
                                 outdir=dataset.dirs[SEMANTIC],
                                 num_workers=4,
                                 batch_size=32)
    

    tokenizer = audiotoken.AudioToken(tokenizer='acoustic', device='cuda:1')
    tokenizer.encode_batch_files(audio_files=files,
                                outdir=dataset.dirs[ACOUSTIC],
                                num_workers=4,
                                batch_size=32)

    # dataset = Dataset(repo_id='mls_eng_10k')
    # tokenizer = get_tokenizer(TEXT, device='cpu')
    # for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
    #     tokens = tokenizer.encode(item.raw_text)  
    #     token_path = dataset.get_absolute_path(item.text_tokens)
    #     np.save(token_path, tokens)

make_dataset()
# tokenize()
