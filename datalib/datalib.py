import json

from dataclasses import dataclass, asdict
from datalib.tokenlib import TEXT, SEMANTIC, ACOUSTIC, AUDIO, ANNOTATIONS, TOKENS
import tarfile
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import upload_file, create_repo

import torchaudio
from tts.utils import convert_audio
from torio.io import CodecConfig
import torch


@dataclass
class Sample:
    id: str = None
    raw_text: str = None
    speaker_id: str = None
    duration: float = None

    audio_path: str = None
    semantic_tokens: str = None
    acoustic_tokens: str = None
    text_tokens: str = None

    metadata: dict = None

    def from_json(self, jss):
        self.__dict__ = json.loads(jss)
        return self

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


class Dataset:
    def __init__(self,
                 repo_id,
                 base_path=Path.home() / '.cache/indri/', 
                 audio_format='.wav'):
    
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)

        self.audio_format = audio_format

        self.repo_id = repo_id
        self.local_path = base_path / self.repo_id
        self.local_path.mkdir(exist_ok=True)

        self.dirs = {
                        ACOUSTIC: self.local_path / TOKENS / ACOUSTIC,
                        SEMANTIC: self.local_path / TOKENS / SEMANTIC,
                        TEXT: self.local_path / TOKENS / TEXT,
                        AUDIO: self.local_path / AUDIO,
                        ANNOTATIONS: self.local_path / ANNOTATIONS
                    }

        for dir in self.dirs.values():
            print("dir=", dir)
            Path(dir).mkdir(exist_ok=True, parents=True)

        self.metadata_path = self.local_path / ANNOTATIONS / 'metadata.jsonl'
        self.metadata_writer = None

        self.hf_token = os.getenv('HF_TOKEN')
        self.hf_user = 'cmeraki'
        
        self.ids = None

    def download(self, hf_repo_id=None, dirs=[ACOUSTIC, SEMANTIC, AUDIO, ANNOTATIONS]):
        if hf_repo_id is None:
            hf_repo_id = self.repo_id

        for name in dirs:
            tar_name = f'{name}.tar'
            hf_hub_download(repo_id=f'{self.hf_user}/{hf_repo_id}',
                            token=self.hf_token,
                            repo_type="dataset",
                            local_dir=self.local_path,
                            filename=tar_name)

        for name in dirs:
            tar_fname = self.local_path / f'{name}.tar'
            print(tar_fname)
            if not tar_fname.exists():
                print("Does not exist", tar_fname)
                continue
            
            tf = tarfile.open(tar_fname)
            tf.extractall(path=self.local_path)
            tf.close()
            # print("Deleting", tar_fname)
            # os.remove(tar_fname)

    def upload(self, hf_repo_id=None):
        if hf_repo_id is None:
            hf_repo_id = self.repo_id

        print(f'Creating repo on HuggingFace, repo_id: {self.hf_user}/{hf_repo_id}')
        create_repo(repo_id=f'{self.hf_user}/{hf_repo_id}',
                    repo_type="dataset",
                    token=self.hf_token,
                    exist_ok=True)

        if hf_repo_id is None:
            hf_repo_id = self.repo_id

        for name in self.dirs:
            if name == 'audio':
                print('skipping audio')
                continue
            dir = self.dirs[name]
            print(f'archiving {name}:{dir}')

            tar_fname = self.local_path / f'{name}.tar'
            arcname = dir.relative_to(self.local_path)
            with tarfile.open(tar_fname, "w") as tar:
                tar.add(dir, arcname=arcname)

            print(f'uploading {name}:{tar_fname}')
            upload_file(repo_id=f'{self.hf_user}/{hf_repo_id}',
                        repo_type="dataset",
                        path_or_fileobj=tar_fname,
                        path_in_repo=f'{name}.tar',
                        token=self.hf_token)

            print("Deleting", tar_fname)
            os.remove(tar_fname)

    @staticmethod
    def create_sample(id, audio_format):
        sample = Sample()
        sample.id = id
        sample.audio_path = str(f'{AUDIO}/{id}{audio_format}')
        sample.semantic_tokens = str(f'{TOKENS}/{SEMANTIC}/{id}.npy')
        sample.acoustic_tokens = str(f'{TOKENS}/{ACOUSTIC}/{id}.npy')
        sample.text_tokens = str(f'{TOKENS}/{TEXT}/{id}.npy')
        return sample

    def get_absolute_path(self, path: str):
        return self.local_path / path

    def add_metadata(self, sample: Sample):
        if self.metadata_writer is None:
            self.metadata_writer = open(self.metadata_path, 'a')
        
        sample.audio_array = None
        sample = sample.to_json()
        self.metadata_writer.write(sample + '\n')
        self.metadata_writer.flush()
    
    def add_audio(self, sample: Sample):
        audio_array = torch.tensor(sample.audio_array, dtype=torch.float32)
        audio_array = audio_array.unsqueeze(dim=0)

        audio_array = convert_audio(
            audio_array,
            sr=sample.sampling_rate,
            target_sr=16000,
            target_channels=1
        )

        torchaudio.save(
            self.get_absolute_path(sample.audio_path),
            audio_array,
            sample_rate=16000,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )

    def add_sample(self, sample: Sample):
        self.add_audio(sample)
        self.add_metadata(sample)

    def iter_dataset(self):
        metadata_path = self.metadata_path
        if not self.metadata_path.exists():
            return 
        
        with open(metadata_path) as metadata:
            for line in metadata:
                print(line)
                sample = Sample().from_json(line)
                yield sample

    def has(self, id):
        if self.ids is None:
            self.ids = []
            for sample in self.iter_dataset():
                self.ids.append(sample.id)
            
        return id in self.ids

    def close(self):
        if self.metadata_writer:
            self.metadata_writer.close()


    def tokenize(self):
        import audiotoken
        from glob import glob
        import numpy as np
        from tqdm import tqdm
        from datalib.tokenlib import (AUDIO,
                                    SEMANTIC,
                                    ACOUSTIC,
                                    TEXT,
                                    get_tokenizer)
        
        dataset = Dataset(repo_id=self.repo_id)
        path = str(dataset.dirs[AUDIO] / f"*{self.audio_format}")
        print(path)
        files = glob(path)

        print("nfiles", len(files))
        print("from", dataset.dirs[AUDIO], "to", dataset.dirs[SEMANTIC])

        tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.semantic_s, device='cuda:0')
        tokenizer.encode_batch_files(audio_files=files,
                                    outdir=dataset.dirs[SEMANTIC],
                                    num_workers=4,
                                    chunk_size=15,
                                    batch_size=32)

        tokenizer = audiotoken.AudioToken(tokenizer=audiotoken.Tokenizers.acoustic, device='cuda:0', num_codebooks=2)
        tokenizer.encode_batch_files(audio_files=files,
                                    outdir=dataset.dirs[ACOUSTIC],
                                    chunk_size=15,
                                    num_workers=4,
                                    batch_size=32)


        tokenizer = get_tokenizer(TEXT, device='cpu')
        for item in tqdm(dataset.iter_dataset(), desc='iterating...'):
            tokens = tokenizer.encode(item.raw_text)
            token_path = dataset.get_absolute_path(item.text_tokens)
            np.save(token_path, tokens)



def create_tar(dir, tar_path):
    tar_cmd = [
        'tar',
        '-cf',
        tar_path,
        dir]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='prepares the data for consumption')
    parser.add_argument('--dataset', type=str, required=True, help='Input directory for audio files.')

    args = parser.parse_args()
    dataset = Dataset(repo_id=args.dataset)
    dataset.download(audio=True)
    for elem in dataset.iter_dataset():
        print(elem)
