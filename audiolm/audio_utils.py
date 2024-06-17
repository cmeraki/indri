import torchaudio
from encodec.utils import convert_audio
import torch
def read_audio_file(fpath, sample_rate):
    waveform, sr = torchaudio.load(fpath)
    waveform = convert_audio(waveform,
                             sr,
                             target_sr=sample_rate,
                             target_channels=1)
    return waveform

def pad_batch(waveforms):
    sizes = [waveform.size(1) for waveform in waveforms]
    max_length = max(sizes)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (-1, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sizes
