import tarfile
import glob
import os
from utils import Sample
from pathlib import Path
import pandas as pd
from tokenlib import encode_files, TEXT


def get_flag(path):
    return (path / 'SUCCESS').exists()


def set_flag(path):
    with open(path / 'SUCCESS', 'w') as flag:
        flag.write('<COMPLETE>')


def get_token_dataset():
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id='cmeraki/gsxl_tokens',
                             repo_type='dataset',
                             token=os.getenv('HF_TOKEN_CMERAKI'))

    path = Path(path)
    print("Dataset path:", path)

    if get_flag(path):
        print("Untar already completed, skipping")
        return path

    for tarname in glob.iglob(str(path / "*.tar")):
        print("Extracting", tarname)
        tf = tarfile.open(path / tarname)
        tf.extractall(path=path)
        tf.close()
        print("Deleting", tarname)
        os.remove(path / tarname)

    set_flag(path)

    return path


def iter_dataset(path):
    metadata_path = path / 'metadata.csv'
    df_metadata = pd.read_csv(metadata_path)

    columns = ['sid', 'text_tn']
    df_metadata = df_metadata[columns].to_dict(orient='records')

    idx = 0
    for example in df_metadata:

        idx += 1
        try:
            token_path = lambda x: path / x / f"{example['sid']}.npy"
            sample = Sample(text=example['text_tn'].lower(), id=example['sid'])
            yield sample
        except Exception as e:
            print("errors", e)


def prepare_data(path):
    dataset = iter_dataset(path)
    encode_files(dataset, path / TEXT, TEXT, device='cpu')