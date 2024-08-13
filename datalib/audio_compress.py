import subprocess
import os
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool


def reduce_audio_bitrate(f):
    target_bitrate = '64k'
    input_file, output_file = f
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-v', '0',
        '-i', input_file,
        '-acodec', 'libmp3lame',
        '-b:a', target_bitrate,
        output_file
    ]

    subprocess.run(ffmpeg_cmd, check=True)


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

    with Pool(processes=8) as pool:
        list(tqdm(pool.imap(reduce_audio_bitrate, file_list), total=len(file_list), desc="Processing files"))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inp', required=True)
    parser.add_argument('--out', required=True)

    args = parser.parse_args()

    compress_dir(args.inp, args.out)


