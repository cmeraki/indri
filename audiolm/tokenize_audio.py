import numpy as np
from encodec.utils import convert_audio
from pathlib import Path

import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

DEVICE = 'cuda:0'


class AudioDataset(Dataset):
    def __init__(self, audio_files, sample_rate, tokenizer):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        waveform = convert_audio(waveform,
                                 sr,
                                 target_sr=self.sample_rate,
                                 target_channels=1)

        tokens = self.tokenizer.encode(waveform)
        return tokens, audio_path


def collate_fn(batch):
    tokens, audio_path = zip(*batch)
    return tokens, audio_path


@torch.inference_mode()
def encode(files, outdir, type='acoustic'):
    """
    Create large files in outdir with memmaps
    of shape [N, C, S].
    :param audio:
    :param out:
    :param batch_size:
    :return:
    """

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.amp.autocast(device_type='cuda', dtype='bfloat16')

    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    tokenizer = None
    from audio_tokenizers import HubertTokenizer, EncodecTokenizer, SEMANTIC, ACOUSTIC
    if type == SEMANTIC:
        tokenizer = HubertTokenizer(device=DEVICE)
    if type == ACOUSTIC:
        tokenizer = EncodecTokenizer(n_codebooks=8, device=DEVICE)

    dataset = AudioDataset(files, sample_rate=tokenizer.audio_sample_rate, tokenizer=tokenizer)

    for file_index in tqdm(range(len(dataset))):
        tokens, path = dataset[file_index]
        path = Path(path)
        fname = path.name
        np.save(outdir / fname, tokens)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--type', type=str, required=True, help='Type of token semantic/acoustic')

    args = parser.parse_args()

    from audio_utils import find_audio_files
    # files = list(glob.glob(args.indir))
    files = find_audio_files(args.indir)
    # print(files)
    # print(random.sample(files, 2))
    encode(files=files, outdir=args.outdir, type=args.type)