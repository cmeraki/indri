import numpy as np
from encodec.utils import convert_audio
from pathlib import Path

import torchaudio
import torch
from tqdm import tqdm

from audio_tokenizers import HubertTokenizer, EncodecTokenizer, TextTokenizer
from audio_tokenizers import SEMANTIC, ACOUSTIC, TEXT
from transformers import AutoTokenizer

DEVICE = 'cuda:0'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.amp.autocast(device_type='cuda', dtype='bfloat16')


def get_tokenizer(type):
    tokenizer = None
    if type == SEMANTIC:
        tokenizer = HubertTokenizer(device=DEVICE)

    if type == ACOUSTIC:
        tokenizer = EncodecTokenizer(n_codebooks=8, device=DEVICE)

    if type == TEXT:
        tokenizer = TextTokenizer()

    return tokenizer


@torch.inference_mode()
def encode_audio_files(files, outdir, type='acoustic'):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    tokenizer = get_tokenizer(type)

    for file in tqdm(files):
        path = Path(file)
        fname = path.name
        outpath = outdir / fname

        if outpath.exists():
            # Dont process files twice
            continue

        try:
            # Sometimes audio files are empty
            # Sometimes they are corrupt
            waveform, sr = torchaudio.load(file)
            waveform = convert_audio(waveform,
                                     sr,
                                     target_sr=tokenizer.audio_sample_rate,
                                     target_channels=1)

            tokens = tokenizer.encode(waveform)

            np.save(outpath, tokens)

        except:
            print(f"Error processing : {file}")


def get_audio_file_paths():
    from datasets import load_dataset
    gs = load_dataset("speechcolab/gigaspeech",
                      "s",
                      token='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')
    files = []
    splits = ["train"]
    for split in splits:
        for example in gs[split]:
            audio_input = example["audio"]['path']
            files.append(audio_input)

    return files


def get_text(outdir):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    from datasets import load_dataset
    gs = load_dataset("speechcolab/gigaspeech",
                      "xs",
                      token='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')

    tokenizer = AutoTokenizer.from_pretrained('TheBloke/Llama-2-7B-Chat-GPTQ')

    texts = {}
    splits = ["train"]
    for split in splits:
        for example in tqdm(gs[split]):
            text = example['text']
            segment_id = example['segment_id']
            texts[segment_id] = text
            tokens = tokenizer.encode(text)
            outpath = outdir / segment_id
            tokens = np.asarray(tokens, dtype=np.uint32)
            np.save(outpath, tokens)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--indir', type=str, required=True, help='Input directory for audio files.')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory for encoded audio.')
    parser.add_argument('--type', type=str, required=True, help='Type of token semantic/acoustic')

    args = parser.parse_args()

    from audio_utils import find_audio_files

    get_text(args.outdir)

    # files = find_audio_files(args.indir)
    # files = get_audio_file_paths()
    # print(len(files))
    # encode_audio_files(files=files, outdir=args.outdir, type=args.type)
