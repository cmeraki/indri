import torch
import torchaudio
from datasets import load_dataset
from tts.utils import convert_audio
from datalib.datalib import Dataset
from torio.io import CodecConfig
from tqdm import tqdm
from datalib.tokenlib import AUDIO, get_tokenizer, SEMANTIC, ACOUSTIC, TEXT


def stream_samples(hf_repo_id):
    dataset = load_dataset(hf_repo_id,
                           split='train')

    return iter(dataset)


def make_dataset():
    dataset = Dataset(repo_id='jenny')

    for item in tqdm(stream_samples('reach-vb/jenny_tts_dataset'), desc='iterating dataset'):
        sample = dataset.create_sample(id=item['file_name'].replace('/', '_'))

        sample.raw_text = item['transcription_normalised']
        sample.speaker_id = 'jenny'

        audio_array = item['audio']['array']
        audio_array = torch.tensor(audio_array, dtype=torch.float32)
        audio_array = audio_array.unsqueeze(dim=0)

        audio_array = convert_audio(audio_array,
                                    sr=48000,
                                    target_sr=16000,
                                    target_channels=1)
        
        torchaudio.save(dataset.get_absolute_path(sample.audio_path),
                        audio_array,
                        sample_rate=16000,
                        format='mp3',
                        encoding='PCM_S',
                        bits_per_sample=16,
                        backend='ffmpeg',
                        compression=CodecConfig(bit_rate=64000)
                        )

        dataset.add_sample(sample)


def tokenize():
    import audiotoken
    import os
    from datalib.tokenlib import AUDIO
    import numpy as np
    
    dataset = Dataset(repo_id='jenny')
    from glob import glob
    path = str(dataset.dirs[AUDIO] / "*.wav")
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
    

    tokenizer = audiotoken.AudioToken(tokenizer='acoustic', device='cuda:0')
    tokenizer.encode_batch_files(audio_files=files,
                                outdir=dataset.dirs[ACOUSTIC],
                                num_workers=4,
                                batch_size=32)

    dataset = Dataset(repo_id='jenny')
    tokenizer = get_tokenizer(TEXT, device='cpu')
    for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
        tokens = tokenizer.encode(item.raw_text)  
        token_path = dataset.get_absolute_path(item.text_tokens)
        np.save(token_path, tokens)



def test_dataset():
    dataset = Dataset(repo_id='jenny')
    for item in tqdm(dataset.iter_dataset()):
        pass

def upload_dataset():
    dataset = Dataset(repo_id='jenny')
    dataset.upload()

make_dataset()
tokenize()
test_dataset()
upload_dataset()

