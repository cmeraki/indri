import math

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

DEVICE = 'cpu'
START_TOKEN = 0

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


def flatten_codebook(arr):
    # give a batch of audio tokens to flatten
    # new_tokenid = old_tokenid + 1024 * codebook_idx
    assert len(arr.shape) == 3
    assert arr.shape[1] < 8

    b, c, n = arr.shape
    i_values = np.arange(c) * 1024
    arr += i_values.reshape(1, c, 1)
    flat_arr = arr.reshape(b, c*n, order='F')
    return flat_arr


def add_start_tokens(arr, dtype):
    start_token = -1
    b = arr.shape[0]
    start_tokens = [start_token] * b
    start_tokens = np.asarray(start_tokens, dtype=dtype)
    start_tokens = np.expand_dims(start_tokens, axis=-1)
    arr = np.hstack([arr, start_tokens])
    return arr


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
        with torch.no_grad():
            encoded_frames = model.encode(batch)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.detach().cpu().numpy().astype(dtype=dtype)

        expected_lengths = np.ceil(np.asarray(sizes)/320).astype(int)

        codes = codes[:, 0:2, :]

        for code, size in zip(codes, expected_lengths):
            code[:, size:] = -2
            code[:, size-1] = START_TOKEN

        codes = flatten_codebook(codes)
        codes = codes.reshape(-1)
        codes = codes[codes != -2]

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

    files = list(glob.glob(args.indir))
    encode(files, outdir=args.outdir, batch_size=args.batch_size)