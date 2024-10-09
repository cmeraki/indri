import subprocess
import os
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import torchaudio
from torio.io import CodecConfig

def reduce_audio_bitrate(f):
    try:
        input_file, output_file = f
        audio_array, sr = torchaudio.load(input_file)
        audio_array = audio_array.to('cuda:0')
        audio_array = torchaudio.functional.resample(audio_array, orig_freq=sr, new_freq=24000)
        audio_array = audio_array.cpu()
        torchaudio.save(
            output_file,
            audio_array,
            sample_rate=24000,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
    except:
        pass

def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files


def compress_dir(input_dir, output_dir):
    files = find_audio_files(input_dir)

    made_dirs = set()
    output_filepaths = []

    for f in tqdm(files, desc='making dirs'):
        out = f.replace(input_dir, output_dir)
        out = Path(out).with_suffix('.wav')
        output_filepaths.append(out)
        p = Path(out).parents[0]
        if p not in made_dirs:
            os.makedirs(p, exist_ok=True)
            made_dirs.add(p)

    file_list = list(zip(files, output_filepaths))

    with Pool(processes=32) as pool:
        list(tqdm(pool.imap(reduce_audio_bitrate, file_list), total=len(file_list), desc="Processing files"))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inp', required=True)
    parser.add_argument('--out', required=True)

    args = parser.parse_args()

    compress_dir(args.inp, args.out)


