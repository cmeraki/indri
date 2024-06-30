import torchaudio
from encodec.utils import convert_audio, save_audio
import torch
import os
from dataclasses import dataclass


@dataclass
class Sample:
    audio_path: str = None
    text: str = None
    id: str = None


def iter_dataset(repo, name, splits):
    from datasets import load_dataset
    gs = load_dataset(repo, name, token='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')

    for split in splits:
        for example in gs[split]:
            Sample(audio_path=example["audio"]["path"],
                   text=example["text"],
                    id=example["segment_id"]
                   )
            yield example


def read_audio_file(fpath, sample_rate):
    waveform, sr = torchaudio.load(fpath)
    waveform = convert_audio(waveform,
                             sr,
                             target_sr=sample_rate,
                             target_channels=1)
    return waveform


def find_audio_files(folder):
    # Define the file extensions to search for
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')

    # List to store the paths of found audio files
    audio_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files

