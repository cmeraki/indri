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


def process_flac_files_in_tar(tar_path):
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.flac'):
                file_content = tar.extractfile(member)
                if file_content is not None:
                    buffer = io.BytesIO(file_content.read())
                    waveform, sample_rate = torchaudio.load(buffer, format="flac")
                    yield member.name, waveform, sample_rate


def stream_samples():
    local_path = Path('/media/apurva/HD-PCTU3/indri_data/peoples_speech/train')
    meta = dict()

    with open(local_path / 'clean.json', "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading metadata file"):
            sample_meta = json.loads(line)
            _id = sample_meta["audio_document_id"]
            texts = sample_meta["training_data"]["label"]
            audio_filenames = sample_meta["training_data"]["name"]
            durations = sample_meta["training_data"]["duration_ms"]
            for audio_filename, text, duration in zip(audio_filenames, texts, durations):
                audio_filename = audio_filename.lstrip("./")
                meta[audio_filename] = {
                    "audio_document_id": _id,
                    "text": text,
                    "duration_ms": duration
                }

    print("Total samples", len(meta))

    for tarname in sorted((local_path/'clean').glob('*.tar'))[::-1]:
        for (name, waveform, sample_rate) in process_flac_files_in_tar(tarname):
            item = {
                "id": name,
                "audio": waveform,
                "sample_rate": sample_rate,
                "text": meta[name]["text"],
                "duration_ms": meta[name]["duration_ms"]
            }

            yield item

@torch.inference_mode()
def make_dataset():
    tokenizers = {name: get_tokenizer(name, device='cuda:1') for name in [ACOUSTIC, SEMANTIC]}
    
    dataset = Dataset(repo_id='peoples_speech')
    for item in tqdm(stream_samples()):
        id = item['id']
        if dataset.has(id):
            # print(f"skipping id:{id}")
            continue
        
        sample = dataset.create_sample(id=id)
        sample.raw_text = item['text']
        sample.speaker_id = ''
        sample_rate = item['sample_rate']

        audio_array = item['audio']
        # tokens = {}
        # for tokenizer_name in tokenizers:
        #     tokenizer = tokenizers[tokenizer_name]
        #     _array = convert_audio(audio_array,
        #                                 sr=tokenizer.audio_sample_rate,
        #                                 target_sr=16000,
        #                                 target_channels=1)
            
        #     tokens[tokenizer_name] = tokenizer.encode(_array).astype(np.int16)
        
        # np.save(dataset.get_absolute_path(sample.semantic_tokens), tokens[SEMANTIC])
        # np.save(dataset.get_absolute_path(sample.acoustic_tokens), tokens[ACOUSTIC])
        
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

    
    # dataset.upload(hf_repo_id='peoples_speech')


def tokenize():
    import audiotoken
    from datalib.tokenlib import AUDIO
    
    dataset = Dataset(repo_id='peoples_speech')
    from glob import glob
    path = str(dataset.dirs[AUDIO] / "*.wav")
    print(path)
    files = glob(path)
    print("nfiles", len(files))
    print("from", dataset.dirs[AUDIO], "to", dataset.dirs[SEMANTIC])

    tokenizer = audiotoken.AudioToken(tokenizer='semantic_s', device='cuda:0')
    tokenizer.encode_batch_files(audio_files=files,
                                 outdir=dataset.dirs[SEMANTIC],
                                 num_workers=4,
                                 batch_size=32)
    

    # tokenizer = audiotoken.AudioToken(tokenizer='acoustic', device='cuda:0', compile=True)
    # tokenizer.encode_batch_files(audio_files=files,
    #                             outdir=dataset.dirs[ACOUSTIC],
    #                             num_workers=4,
    #                             chunk_size=15,
    #                             batch_size=128)



    dataset = Dataset(repo_id='peoples_speech')
    tokenizer = get_tokenizer(TEXT, device='cuda:0')
    for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
        tokens = tokenizer.encode(item.raw_text)  
        token_path = dataset.get_absolute_path(item.text_tokens)
        np.save(token_path, tokens)


# def test_dataset():
#     dataset = Dataset(repo_id='peoples_speech')
#     for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
#         pass

# make_dataset()
tokenize()
