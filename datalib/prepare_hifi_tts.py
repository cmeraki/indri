import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset
from torio.io import CodecConfig

from tts.utils import convert_audio
from datalib.datalib import Dataset

def stream_samples(hf_repo_id):
    dataset = load_dataset(
        hf_repo_id,
        'all',
        split='train.other'
    )

    return iter(dataset)

def make_dataset(repo_id):
    dataset = Dataset(repo_id=repo_id)

    for item in tqdm(stream_samples('MikhailT/hifi-tts')):
        sample = dataset.create_sample(id=item['file'].replace('/', '__'))
        sample.raw_text = item['text_no_preprocessing']
        sample.speaker_id = item['speaker']
        sample.duration = item['duration']

        audio_array = item['audio']['array']
        audio_array = torch.tensor(audio_array, dtype=torch.float32)
        audio_array = audio_array.unsqueeze(dim=0)

        sampling_rate = item['audio']['sampling_rate']

        audio_array = convert_audio(
            audio_array,
            sr=sampling_rate,
            target_sr=16000,
            target_channels=1
        )

        torchaudio.save(
            dataset.get_absolute_path(sample.audio_path),
            audio_array,
            sample_rate=16000,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )

        dataset.add_sample(sample)

def tokenize(repo_id):
    import audiotoken
    import numpy as np
    from glob import glob

    from datalib.tokenlib import (AUDIO, SEMANTIC, ACOUSTIC, TEXT, get_tokenizer)

    dataset = Dataset(repo_id=repo_id)
    path = str(dataset.dirs[AUDIO] / "*.wav")
    
    print(path)
    
    files = glob(path)

    print("nfiles", len(files))
    print("from", dataset.dirs[AUDIO], "to", dataset.dirs[SEMANTIC])

    tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.semantic_s, device='cuda:0')
    tokenizer.encode_batch_files(
        audio_files=files,
        outdir=dataset.dirs[SEMANTIC],
        num_workers=4,
        batch_size=32
    )

    tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.acoustic, device='cuda:0')
    tokenizer.encode_batch_files(
        audio_files=files,
        outdir=dataset.dirs[ACOUSTIC],
        num_workers=4,
        batch_size=32
    )

    tokenizer = get_tokenizer(TEXT, device='cpu')

    for item in tqdm(dataset.iter_dataset(), desc='Tokenizing text...'):
        tokens = tokenizer.encode(item.raw_text)
        token_path = dataset.get_absolute_path(item.text_tokens)
        np.save(token_path, tokens)

def test_dataset(repo_id):
    dataset = Dataset(repo_id=repo_id)
    for item in tqdm(dataset.iter_dataset()):
        pass

make_dataset('hifi_tts_other')
test_dataset('hifi_tts_other')
tokenize('hifi_tts_other')
ds = Dataset('hifi_tts_other')
ds.upload()

