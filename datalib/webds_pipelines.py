from create_dataset_from_audio import Transcriber
from torch.utils.data import DataLoader
import webdataset as wds
from pathlib import Path
import os
import torchaudio

def break_audio_chunks(input_ds, output_dir, output_ds):
    hf_token = os.environ['HF_TOKEN']
    url = "https://huggingface.co/datasets/cmeraki/youtube_webdataset/resolve/main/en__storiesofmahabharata__{{000000..000000}}.tar"
    url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"

    cache_dir = Path('~/.cache/wds/tmp/').expanduser()
    os.makedirs(cache_dir, exist_ok=True)

    def get_sample(item):
        txt = item['json']['raw_text']
        audio = item['wav']

        return txt, audio
    
    transcriber = Transcriber(device='cuda:0')

    dataset = wds.WebDataset(url, shardshuffle=None, cache_dir=cache_dir).decode()
    
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

break_audio_chunks(1,1,1)