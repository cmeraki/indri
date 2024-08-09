import json

from dataclasses import dataclass, asdict
from audiolm.tokenlib import TEXT, SEMANTIC, ACOUSTIC, AUDIO, ANNOTATIONS, TOKENS
import tarfile
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub import upload_file, create_repo


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

    def from_json(self, jss):
        self.__dict__ = json.loads(jss)
        return self

    def to_json(self):
        return json.dumps(asdict(self))


class Dataset:
    def __init__(self,
                 repo_id,
                 base_path=Path.home() / '.cache/indri/'):

        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)

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

        self.hf_token = os.getenv('HF_TOKEN_CMERAKI')
        self.hf_user = 'cmeraki'

    def download(self, hf_repo_id=None):
        if hf_repo_id is None:
            hf_repo_id = self.repo_id

        for name in self.dirs:
            tar_name = f'{name}.tar'
            hf_hub_download(repo_id=f'{self.hf_user}/{hf_repo_id}',
                            token=self.hf_token,
                            local_dir=self.local_path,
                            filename=tar_name)

        for name in self.dirs:
            tar_fname = self.local_path / f'{name}.tar'
            print(tar_fname)
            tf = tarfile.open(tar_fname)
            tf.extractall(path=self.local_path)
            tf.close()
            # print("Deleting", tar_fname)
            # os.remove(tar_fname)

    def upload(self, hf_repo_id=None):
        create_repo(repo_id=f'{self.hf_user}/{hf_repo_id}',
                    token=self.hf_token,
                    exist_ok=True)

        if hf_repo_id is None:
            hf_repo_id = self.repo_id

        for name in self.dirs:
            dir = self.dirs[name]
            print('archiving {name}:{dir}')

            tar_fname = self.local_path / f'{name}.tar'
            arcname = dir.relative_to(self.local_path)
            with tarfile.open(tar_fname, "w") as tar:
                tar.add(dir, arcname=arcname)

            upload_file(repo_id=f'{self.hf_user}/{hf_repo_id}',
                        path_or_fileobj=tar_fname,
                        path_in_repo=f'{name}.tar',
                        token=self.hf_token)

            print("Deleting", tar_fname)
            os.remove(tar_fname)


    def create_sample(self, id):
        sample = Sample()
        sample.id = id
        sample.audio_path = str(f'{AUDIO}/{id}.wav')
        sample.semantic_tokens = str(f'{TOKENS}/{SEMANTIC}/{id}.npy')
        sample.acoustic_tokens = str(f'{TOKENS}/{ACOUSTIC}/{id}.npy')
        sample.text_tokens = str(f'{TOKENS}/{TEXT}/{id}.npy')
        return sample

    def get_absolute_path(self, path: str):
        return self.local_path / path

    def add_sample(self, sample: Sample):
        if self.metadata_writer is None:
            self.metadata_writer = open(self.metadata_path, 'a')

        sample = sample.to_json()
        self.metadata_writer.write(sample + '\n')
        self.metadata_writer.flush()

    def iter_dataset(self):
        metadata_path = self.metadata_path
        with open(metadata_path) as metadata:
            for line in metadata:
                sample = Sample().from_json(line)
                yield sample

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
    dataset.download()
    for elem in dataset.iter_dataset():
        print(elem)
