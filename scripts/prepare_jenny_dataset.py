import torch
import torchaudio
from datasets import load_dataset
from audiolm.utils import convert_audio
from audiolm.datalib import Dataset
from torio.io import CodecConfig
from tqdm import tqdm
from audiolm.tokenlib import AUDIO 


def stream_samples(hf_repo_id):
    dataset = load_dataset(hf_repo_id,
                           split='train',
                           streaming=True)

    return iter(dataset)


def make_dataset():
    dataset = Dataset(repo_id='jenny')

    for item in tqdm(stream_samples('reach-vb/jenny_tts_dataset')):
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

    dataset.upload(hf_repo_id='jenny')


def tokenize():
    dataset = Dataset(repo_id='jenny')
    
    import audiotoken
    tokenizer = audiotoken.AudioToken(tokenizer='semantic_s', device='cuda:0')
    print(dataset.dirs[AUDIO])
    tokenizer.encode_batch_files(audio_dir=dataset.dirs[AUDIO],
                                 outdir='/tmp/test/',
                                 num_workers=4,
                                 batch_size=32)


def test_dataset():
    dataset = Dataset(repo_id='jenny')
    for item in tqdm(dataset.iter_dataset()):
        pass


# make_dataset()
tokenize()
test_dataset()

