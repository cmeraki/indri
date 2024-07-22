import os
import tarfile
import multiprocessing
from functools import partial
from tqdm import tqdm
import subprocess


def create_tar(args):
    batch, tar_number, output_dir = args
    tar_path = os.path.join(output_dir, f"archive_{tar_number}.tar")

    tar_cmd = [
        'tar',
        '-cf',
        tar_path
    ] + batch

    subprocess.run(tar_cmd, check=True)


def create_tars(source_dir, output_dir, size_limit_gb=1, num_cores=1):
    size_limit = size_limit_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    current_size = 0
    tar_count = 1
    current_batch = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare batches
    batches = []
    for root, _, files in tqdm(os.walk(source_dir), desc='preparing batches'):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)

            if current_size + file_size > size_limit:
                batches.append((current_batch, tar_count, output_dir))
                tar_count += 1
                current_size = 0
                current_batch = []

            current_batch.append(file_path)
            current_size += file_size

    # Add the last batch if it's not empty
    if current_batch:
        batches.append((current_batch, tar_count,  output_dir))

    # Use all available cores if num_cores is not specified
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

    print("Starting to tar files ..")

    # Create tar files in parallel
    with multiprocessing.Pool(num_cores) as pool:
        list(tqdm(pool.imap(create_tar, batches), total=len(batches), desc="Processing files"))

    print(f"Process completed. Created {len(batches)} tar files.")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--inp', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--size', required=False, default=1, type=float)
    parser.add_argument('--cores', required=False, default=1, type=int)

    args = parser.parse_args()
    create_tars(args.inp, args.out, size_limit_gb=args.size, num_cores=args.cores)