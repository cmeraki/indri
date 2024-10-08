from tqdm import tqdm
from glob import glob
from pathlib import Path
import webdataset as wds
from datasets import load_dataset
from huggingface_hub import upload_file

from datalib.mappings import dataset_info

def iter_hf_item(dsname, streaming=False):
    dinfo = dataset_info[dsname]
    dconfig = {k: dinfo[k] for k in ['path', 'split', 'name']}

    dataset = load_dataset(
        streaming=streaming,
        **dconfig
    )

    dataset = iter(dataset)

    for idx, item in tqdm(enumerate(dataset)):
        yield item


def generate_wds_samples(dsname):
    for item in iter_hf_item(dsname):
        transform_func = dataset_info[dsname]['method']
        sample = transform_func(item)

        yield sample

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Create and upload a webdataset from huggingface dataset')
    parser.add_argument('--dsname', type=str, required=True, help='Name of your dataset. Needs to have a registered mapping function.')
    parser.add_argument('--outprefix', type=str, required=True, help='Prefix for the webdataset shards')
    parser.add_argument('--cache_dir', default='~/.cache/wds/prepare/', type=str, required=False, help='Path to cache the webdataset shards while transforming')

    args = parser.parse_args()

    hf_user = 'cmeraki'
    hf_repo = 'audiofolder_webdataset'
    hf_token = os.environ['CMERAKI_HF_TOKEN']
    cache_dir = Path(args.cache_dir).expanduser()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(f"{Path(cache_dir, args.outprefix)}__%06d.tar") as sink:
        for sample in generate_wds_samples(args.dsname):
            sink.write(sample)

    tar_files = glob(f"{cache_dir}/{args.outprefix}__*.tar")
    print(f'Uploading {args.dsname} to huggingface with {len(tar_files)} shards')

    for tar_file in tqdm(tar_files, desc='Uploading shards'):
        upload_file(
            repo_id=f'{hf_user}/{hf_repo}',
            repo_type="dataset",
            path_or_fileobj=tar_file,
            path_in_repo=tar_file.replace(str(cache_dir), ''),
            token=hf_token
        )

    print(f'Uploaded {args.dsname} to huggingface')
