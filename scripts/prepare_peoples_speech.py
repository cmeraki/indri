import json
import tarfile
import io
import torchaudio

from torio.io import CodecConfig
from tqdm import tqdm
from pathlib import Path

from audiolm.utils import convert_audio
from audiolm.datalib import Dataset


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
    local_path = Path('/mnt/HD-PCTU3/indri_data/peoples_speech/train')
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

    for tarname in sorted((local_path/'clean').glob('*.tar')):
        for (name, waveform, sample_rate) in process_flac_files_in_tar(tarname):
            item = {
                "id": name,
                "audio": waveform,
                "sample_rate": sample_rate,
                "text": meta[audio_filename]["text"],
                "duration_ms": meta[audio_filename]["duration_ms"]
            }

            yield item


def make_dataset():
    dataset = Dataset(repo_id='peoples_speech')
    for item in tqdm(stream_samples()):
        sample = dataset.create_sample(id=item['id'])
        sample.raw_text = item['text']
        sample.speaker_id = ''
        sample_rate = item['sample_rate']

        audio_array = item['audio']

        audio_array = convert_audio(audio_array,
                                    sr=sample_rate,
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

    dataset.upload(hf_repo_id='peoples_speech')


def test_dataset():
    dataset = Dataset(repo_id='peoples_speech')
    for item in tqdm(dataset.iter_dataset()):
        pass


make_dataset()
test_dataset()

