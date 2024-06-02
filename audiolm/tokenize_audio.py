import os
import math
import random

import numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio, save_audio
from pathlib import Path

import torchaudio
import torch
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import uuid

DEVICE = 'cuda:0'
START_TOKEN = 0


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


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate, channels):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.channels = channels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        waveform = convert_audio(waveform,
                                 sr,
                                 target_sr=self.sample_rate,
                                 target_channels=self.channels)

        return waveform, sr


def collate_fn(batch):
    waveforms, sample_rates = zip(*batch)
    sizes = [waveform.size(1) for waveform in waveforms]
    max_length = max(sizes)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (-1, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sizes


def get_model(model_sr=24, bandwidth=3):
    if model_sr == 24:
        model = EncodecModel.encodec_model_24khz()
    elif model_sr == 48:
        model = EncodecModel.encodec_model_48khz()
    else:
        raise "Unknown model sample rate, chose from 24 and 48"

    model.set_target_bandwidth(bandwidth)
    model.zero_grad()
    return model


def codebook_encoding(arr):
    c, n = arr.shape
    i_values = np.arange(c) * 1024
    arr += i_values.reshape(c, 1)
    return arr

def flatten_codebook(arr):
    # give a batch of audio tokens to flatten
    # new_tokenid = old_tokenid + 1024 * codebook_idx
    assert len(arr.shape) == 2
    assert arr.shape[0] < 8

    c, n = arr.shape
    flat_arr = arr.reshape(c*n, order='F')
    return flat_arr


def add_start_token(arr):
    arr = np.insert(arr, 0, START_TOKEN)
    return arr


@torch.inference_mode()
def encode(files, outdir, batch_size=1, per_file_tokens=1000000):
    """
    Create large files in outdir with memmaps
    of shape [N, C, S].
    :param audio:
    :param out:
    :param batch_size:
    :return:
    """
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    dtype = np.int32

    # torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.amp.autocast(device_type='cuda', dtype='bfloat16')

    model = get_model(bandwidth=3)
    model = model.to(DEVICE)
    dataset = AudioDataset(files, sample_rate=24000, channels=1)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=4)

    model = torch.compile(model)

    from npds import NumpyDataset
    ds = NumpyDataset(dir=outdir, samples_per_file=per_file_tokens, dtype=dtype)

    for batch_index, (batch, sizes) in enumerate(tqdm(dataloader)):
        batch = batch.to(DEVICE)
        encoded_frames = model.encode(batch)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.detach().cpu().numpy().astype(dtype=dtype)

        expected_lengths = np.ceil(np.asarray(sizes)/320).astype(int)

        codes = codes[:, 0:2, :]
        new_codes = []
        for code, size in zip(codes, expected_lengths):
            code = code[:, :size]
            code = codebook_encoding(code)
            code = flatten_codebook(code)
            code = add_start_token(code)
            new_codes.append(code)

        codes = np.hstack(new_codes)

        ds.write(codes)

    ds.close()


def get_next_filename(dir: Path):
    filename = f"{dir}/{str(uuid.uuid4())}"
    return filename


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for encoding.')

    args = parser.parse_args()

    # files = list(glob.glob(args.indir))
    files = find_audio_files(args.indir)
    print(random.sample(files, 10))
    encode(files, outdir=args.outdir, batch_size=args.batch_size)