import torchaudio
from encodec.utils import convert_audio, save_audio
import torch
import os


def read_audio_file(fpath, sample_rate):
    waveform, sr = torchaudio.load(fpath)
    waveform = convert_audio(waveform,
                             sr,
                             target_sr=sample_rate,
                             target_channels=1)
    return waveform


def pad_batch(waveforms):
    sizes = [waveform.size(0) for waveform in waveforms]
    max_length = max(sizes)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(0)
        padded_waveform = torch.nn.functional.pad(waveform, (-1, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sizes


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
