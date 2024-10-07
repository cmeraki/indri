from tqdm import tqdm
from glob import glob
from pathlib import Path
import webdataset as wds
from datasets import load_dataset
from huggingface_hub import upload_file
import os
import io

from datalib.mappings_local import prepare_local_dataset

def generate_wds_samples(dsname,channel_name,language):
    samples = []
    folder_path="/mnt/disks/audio-data-2/Data/"
    channel_path=os.path.join(folder_path,channel_name)
    audio_path=os.path.join(channel_path,"audio_files_compressed/")
    for file in os.listdir(audio_path):
        filename = os.path.basename(file).replace(".wav","")
        sample = prepare_local_dataset(dsname=dsname,channel_name=channel_name,audio_name=filename,folder_path=folder_path,split='train',language=language)
        samples.append(sample)
    return samples

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Create and upload a webdataset from huggingface dataset')
    parser.add_argument('--dsname', type=str, required=True, help='Name of your dataset. Needs to have a registered mapping function.')
    parser.add_argument('--channel_name',type=str,required=True,help='Enter the channel name')
    parser.add_argument('--language',type=str,required=True,help='Enter the language')
    parser.add_argument('--outprefix', type=str, required=True, help='Prefix for the webdataset shards')
    parser.add_argument('--cache_dir', default='~/.cache/wds/prepare/', type=str, required=False, help='Path to cache the webdataset shards while transforming')

    args = parser.parse_args()

    samples = generate_wds_samples(args.dsname,args.channel_name,args.language)

    hf_user = 'cmeraki'
    hf_repo = 'audiofolder_webdataset'
    hf_token = os.environ['CMERAKI_HF_TOKEN']
    cache_dir = Path(args.cache_dir).expanduser()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    with wds.ShardWriter(f"{Path(cache_dir, args.outprefix)}__%06d.tar") as sink:
        for sample in samples:
            sink.write(sample)

    tar_files = glob(f"{cache_dir}/{args.outprefix}__*.tar")
    print(f'Uploading {args.dsname} to huggingface with {len(tar_files)} shards')

    '''for tar_file in tqdm(tar_files, desc='Uploading shards'):
        upload_file(
            repo_id=f'{hf_user}/{hf_repo}',
            repo_type="dataset",
            path_or_fileobj=tar_file,
            path_in_repo=tar_file.replace(str(cache_dir), ''),
            token=hf_token
        )

    print(f'Uploaded {args.dsname} to huggingface')'''

