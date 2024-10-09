import os
import io
import torch
import torchaudio
import numpy as np


def read_audio_file(fpath, sample_rate):
    waveform, sr = torchaudio.load(fpath)
    waveform = convert_audio(waveform,
                             sr,
                             target_sr=sample_rate,
                             target_channels=1)
    return waveform


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


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


def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]


def audio_to_wav_bytestring(audio_array, sample_rate):
    """
    Convert np audio array to a wav bytestring that can be used to save an audio file.
    """
    audio_array = torch.tensor(audio_array, dtype=torch.float32)
    audio_array = audio_array.unsqueeze(dim=0)

    audio_array = convert_audio(
        audio_array,
        sr=sample_rate,
        target_sr=16000,
        target_channels=1
    )

    buffer = io.BytesIO()

    torchaudio.save(
        buffer,
        audio_array,
        sample_rate=16000,
        format='mp3',
        encoding='PCM_S',
        bits_per_sample=16,
        backend='ffmpeg',
        compression=torchaudio.io.CodecConfig(bit_rate=64000)
    )

    mp3_bytestring = buffer.getvalue()

    return mp3_bytestring
