import numpy as np
from encodec.utils import convert_audio
from pathlib import Path

import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import uuid

DEVICE = 'cuda:0'

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

        return waveform


def collate_fn(batch):
    waveforms = zip(*batch)
    sizes = [waveform.size(1) for waveform in waveforms]
    max_length = max(sizes)

    padded_waveforms = []
    for waveform in waveforms:
        padding = max_length - waveform.size(1)
        padded_waveform = torch.nn.functional.pad(waveform, (-1, padding))
        padded_waveforms.append(padded_waveform)

    padded_waveforms = torch.stack(padded_waveforms)
    return padded_waveforms, sizes


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

    dataset = AudioDataset(files, sample_rate=24000, channels=1)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=4)


    from npds import NumpyDataset
    ds = NumpyDataset(dir=outdir, samples_per_file=per_file_tokens, dtype=dtype)

    for batch_index, (batch, sizes) in enumerate(tqdm(dataloader)):
        batch = batch.to(DEVICE)

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
    # print(files)
    # print(random.sample(files, 2))
    encode(files, outdir=args.outdir, batch_size=args.batch_size)