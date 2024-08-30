import os
import torch
import zipfile
import glob
import math
import torchaudio
from glob import glob
from tqdm import tqdm
from torio.io import CodecConfig
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader

from tts.utils import convert_audio
from datalib.datalib import Dataset
from datalib.tokenlib import AUDIO, SEMANTIC, ACOUSTIC

def iterate_zip(x: os.PathLike):
    with zipfile.ZipFile(x, 'r') as zip_file:
        for file_info in zip_file.infolist():
            if file_info.is_dir():
                continue

            file_content = zip_file.open(file_info.filename)
            file_name = file_info.filename

            if file_content is None:
                continue

            yield file_name, file_content

class FileIterator(IterableDataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = glob(os.path.join(root_dir, '*.zip'))
        for f in self.file_list:
            if 'freesound.zip' in f or 'bbc.zip' in f:
                self.file_list.remove(f)

        print(f'Found {len(self.file_list)} files')
        

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            per_worker = int(math.ceil(len(self.file_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_list))

        for idx in range(iter_start, iter_end):
            file_path = self.file_list[idx]

            for single_file, single_content in iterate_zip(file_path):
                try:
                    waveform, sample_rate = torchaudio.load(single_content)
                    yield {
                        'file_name': single_file,
                        'audio': waveform,
                        'sample_rate': sample_rate
                    }
                except:
                    print(f'Error for {single_file}')
                    continue

def make_dataset(root_dir, dataset_name):
    dataset = Dataset(repo_id=dataset_name)
    dataloader = DataLoader(
        FileIterator(root_dir),
        batch_size=1
    )

    for item in tqdm(dataloader):
        sample = dataset.create_sample(
            id=item['file_name'][0].replace('/', '_').replace('.wav', '')
        )

        sample.raw_text = ''
        sample.speaker_id = ''

        audio_array = item['audio'][0]

        audio_array = convert_audio(
            audio_array,
            sr=item['sample_rate'][0],
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

    return dataset

def tokenize(dataset_name):
    dataset = Dataset(repo_id=dataset_name)
    import audiotoken

    tokenizer = audiotoken.AudioToken(tokenizer='semantic_s', device='cuda:1')
    audio_files = glob(os.path.join(dataset.dirs[AUDIO], '*.wav'))

    print(dataset.dirs[AUDIO], len(audio_files))

    tokenizer.encode_batch_files(
        audio_files=audio_files,
        outdir=dataset.dirs[SEMANTIC],
        num_workers=4,
        batch_size=32
    )

def test_dataset():
    dataset = Dataset(repo_id='wavcaps')
    for item in tqdm(dataset.iter_dataset()):
        pass

ds = make_dataset(
    root_dir='/mnt/8727e128-d9ed-4e5c-8281-7a87831b41b5/audio/wavcaps/',
    dataset_name='wavcaps'
)
test_dataset()
tokenize('wavcaps')