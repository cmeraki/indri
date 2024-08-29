import torch
import torchaudio
from datasets import load_dataset
from tts.utils import convert_audio
from datalib.datalib import Dataset
from torio.io import CodecConfig
from tqdm import tqdm

DSNAME = 'vctk'

def stream_samples(hf_repo_id):
    dataset = load_dataset(hf_repo_id,
                           split='train')

    return iter(dataset)


def make_dataset():
    dataset = Dataset(repo_id=DSNAME)

    for item in tqdm(stream_samples('CSTR-Edinburgh/vctk')):
        sample = dataset.create_sample(id=item['text_id'] + '_' + item['speaker_id'])
        sample.raw_text = item['text']
        sample.speaker_id = item['speaker_id']
        sample.metadata = {k for k in item if k != 'audio'}

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
    from datalib.tokenlib import (AUDIO,
                                  SEMANTIC,
                                  ACOUSTIC,
                                  TEXT,
                                  get_tokenizer)
    import numpy as np

    dataset = Dataset(repo_id=DSNAME)
    from glob import glob
    path = str(dataset.dirs[AUDIO] / "*.wav")
    print(path)
    files = glob(path)

    print("nfiles", len(files))
    print("from", dataset.dirs[AUDIO], "to", dataset.dirs[SEMANTIC])

    tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.semantic_s, device='cuda:0')
    tokenizer.encode_batch_files(audio_files=files,
                                 outdir=dataset.dirs[SEMANTIC],
                                 num_workers=4,
                                 batch_size=32)

    tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.acoustic, device='cuda:0')
    tokenizer.encode_batch_files(audio_files=files,
                                 outdir=dataset.dirs[ACOUSTIC],
                                 num_workers=4,
                                 batch_size=32)


    tokenizer = get_tokenizer(TEXT, device='cpu')
    for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
        tokens = tokenizer.encode(item.raw_text)
        token_path = dataset.get_absolute_path(item.text_tokens)
        np.save(token_path, tokens)

def test_dataset():
    dataset = Dataset(repo_id=DSNAME)
    for item in tqdm(dataset.iter_dataset()):
        pass

def upload_dataset():
    dataset = Dataset(repo_id=DSNAME)
    dataset.upload(DSNAME)

make_dataset()
tokenize()
test_dataset()
upload_dataset()
