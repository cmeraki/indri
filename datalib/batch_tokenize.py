from transformers import MimiModel, AutoFeatureExtractor
import torchaudio
from glob import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from pathlib import Path
import os
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

class AudioDataset(Dataset):
    def __init__(self, input_files):
        self.input_files = input_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        f = self.input_files[idx]
        audio_array, sr = torchaudio.load(f)
        audio_array = torchaudio.functional.resample(audio_array, orig_freq=sr, new_freq=24000)[0]
        return audio_array, f

def pad_batch(batch):
    file_batch = []
    audio_batch = []
    for a, f in batch:
        audio_batch.append(a)
        file_batch.append(f)
    
    audio_batch_padded = pad_sequence(audio_batch, batch_first=True)
    lengths = torch.tensor([len(t) for t in audio_batch])
    padding_mask = None
    return audio_batch_padded, padding_mask, lengths, file_batch

def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files

torch.inference_mode()
def tokenize(audio_dir, token_dir, device):
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.to(device)
    model.eval()
    n_codebooks = 8

    audio_files = find_audio_files(audio_dir)
    token_dir = Path(token_dir)
    ads = AudioDataset(audio_files)
    dataloader = DataLoader(ads, batch_size=2, 
                            num_workers=2, 
                            collate_fn=pad_batch, 
                            pin_memory=True)

    import math

    pbar = tqdm(total=len(audio_files), desc="tokenizing...")
    
    for audio_batch_padded, padding_mask, lengths, file_batch in dataloader:
        audio_batch = audio_batch_padded.unsqueeze(1).to(device, non_blocking=True)
        batched_out = model.encode(audio_batch, padding_mask, num_quantizers=n_codebooks)
        batched_out = batched_out.audio_codes.to(torch.int16).detach().cpu().numpy()
        codes = []
        for idx in range(len(lengths)):
            length = lengths[idx]
            code_length = math.ceil(length / (24000/12.5))
            code = batched_out[idx, :, :code_length]
            audio_file_path = file_batch[idx]
            out_file_name = Path(audio_file_path).with_suffix('.npy').name
            out_file_path = str(token_dir / out_file_name)
            np.save(out_file_path, code)
            
        pbar.update(len(lengths))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--inp', type=str, required=True, help='directory of audio files')
    parser.add_argument('--out', type=str, required=False, default=None, help='name of speaker if known')
    parser.add_argument('--device', type=str, required=True, help='name of device')
    
    args = parser.parse_args()

    tokenize(audio_dir = args.inp, token_dir = args.out, device=args.device)