from create_dataset_from_audio import Transcriber
from torch.utils.data import DataLoader
import webdataset as wds
from pathlib import Path
import os
import torchaudio
from huggingface_hub import list_repo_files

HF_TOKEN = os.environ['HF_TOKEN']

def get_all_tars_in_hf_repo(repo_id, hf_token=HF_TOKEN):
    files = list_repo_files(repo_id=repo_id, token=hf_token, repo_type='dataset')
    return files


def break_audio_chunks(hf_files, output_dir, output_ds):
    hf_token = os.environ['HF_TOKEN']
    urls = []
    for hf_file in hf_files:
        url = f"https://huggingface.co/datasets/cmeraki/youtube_webdataset/resolve/main/{hf_file}"
        url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
        urls.append(url)
    
    cache_dir = Path('~/.cache/wds/tmp/').expanduser()
    os.makedirs(cache_dir, exist_ok=True)

    transcriber = Transcriber(device='cuda:0')

    dataset = wds.WebDataset(urls, shardshuffle=None, cache_dir=cache_dir).decode()
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=4
    )

    for elem in dataloader:
        wav, sr = torchaudio.load(elem['wav'])
        print(elem['json'], wav.shape)
        # audio = elem.get('audio')
        for chunk in transcriber.get_transcript(wav=wav[0]):
            print(chunk)

def whisper_on_hindi():
    files = get_all_tars_in_hf_repo('cmeraki/youtube_webdataset')
    en_files = list(filter(lambda x:x[:2] == 'en', files))
    print(en_files)
    break_audio_chunks(en_files,1,1)

whisper_on_hindi()