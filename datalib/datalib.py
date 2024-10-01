import json

from dataclasses import dataclass, asdict
from datalib.tokenlib import TEXT, MIMI, AUDIO, ANNOTATIONS, TOKENS
import tarfile
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import upload_file, create_repo

from omni.utils import convert_audio
import torch
from datalib.tokenlib import MimiTokenizer
import numpy as np


@dataclass
class Sample:
    id: str = None
    raw_text: str = None
    speaker_id: str = None
    duration: float = None
    mimi_tokens: str = None
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
                 audio_format='.wav',
                 device='cuda:0'):
    
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)

        self.audio_format = audio_format

        self.repo_id = repo_id
        self.local_path = base_path / self.repo_id
        self.local_path.mkdir(exist_ok=True)

        self.dirs = { MIMI: self.local_path / TOKENS / MIMI,
                    ANNOTATIONS: self.local_path / ANNOTATIONS}

        for dir in self.dirs.values():
            print("dir=", dir)
            Path(dir).mkdir(exist_ok=True, parents=True)

        self.metadata_path = self.local_path / ANNOTATIONS / 'metadata.jsonl'
        self.metadata_writer = None

        self.hf_token = os.getenv('HF_TOKEN')
        self.hf_user = 'cmeraki'
        
        self.ids = None

        self.device = device
        self.audio_tokenizer = MimiTokenizer(device=self.device)

    def download(self, hf_repo_id=None, dirs=[MIMI, AUDIO, ANNOTATIONS]):
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
        sample.mimi_tokens = str(f'{TOKENS}/{MIMI}/{id}.npy')
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
        audio_tokens = self.audio_tokenizer.encode(sample.audio_array.astype(np.float32).reshape(-1))
        audio_tokens_path = self.get_absolute_path(sample.mimi_tokens) 
        np.save(audio_tokens_path, audio_tokens)

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
