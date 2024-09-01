import torch
import torchaudio
from datasets import load_dataset
from tts.utils import convert_audio
from datalib.datalib import Dataset
from torio.io import CodecConfig
from tqdm import tqdm


def stream_samples(hf_repo_id):
    dataset = load_dataset(hf_repo_id,
                           split='train')

    return iter(dataset)


def make_dataset():
    dataset = Dataset(repo_id='expresso')

    for item in tqdm(stream_samples('ylacombe/expresso')):
        sample = dataset.create_sample(id=item['id'])
        sample.raw_text = item['text']
        sample.speaker_id = item['speaker_id']

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
    
    dataset.tokenize()
    dataset.upload()


def test_dataset():
    dataset = Dataset(repo_id='expresso')
    for item in tqdm(dataset.iter_dataset()):
        pass

def upload_dataset():
    dataset = Dataset(repo_id='expresso')
    dataset.upload()


make_dataset()
test_dataset()
upload_dataset()
