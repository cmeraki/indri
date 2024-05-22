import math

from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


DEVICE = 'cuda'

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
    max_length = max(waveform.size(1) for waveform in waveforms)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (0, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sample_rates


def get_audio_batches(files, batch_size, sample_rate, channels):
    n_batches = math.ceil(len(files)//batch_size)

    for batch_index in tqdm(range(n_batches)):
        fnames = files[batch_index * batch_size: batch_index * batch_size + batch_size]
        batch = []
        size = math.inf
        for fname in fnames:
            wav, sr = torchaudio.load(fname)
            wav = convert_audio(wav, sr, sample_rate, channels)
            batch.append(wav)
            size = min(size, wav.shape[1])

        batch = [wav[:, 0:size] for wav in batch]
        batch = torch.stack(batch, dim=0)
        yield batch


def get_model():
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6)
    model.zero_grad()
    return model


def encode(files, outdir, batch_size=1):
    model = get_model()
    model = model.to(DEVICE)
    dataset = AudioDataset(files, sample_rate=24000, channels=1)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=4)

    model = torch.compile(model)

    for batch, sample_rates in tqdm(dataloader):
        batch = batch.to(DEVICE)
        with torch.no_grad():
            encoded_frames = model.encode(batch)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)


indir = "/home/apurva/projects/indri/data/indian_languages/valid_audio/kb_data_clean_m4a/*/valid/audio/*.m4a"
files = list(glob.glob(indir))
outdir = "encoded_audio"
encode(files, outdir=outdir, batch_size=32)