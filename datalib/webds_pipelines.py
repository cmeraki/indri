from create_dataset_from_audio import Transcriber
from torch.utils.data import DataLoader
import webdataset as wds
from pathlib import Path
import os
import torchaudio
from huggingface_hub import list_repo_files
from silero_vad import save_audio
import json

HF_TOKEN = os.environ['HF_TOKEN']

def get_all_tars_in_hf_repo(repo_id, hf_token=HF_TOKEN):
    files = list_repo_files(repo_id=repo_id, token=hf_token, repo_type='dataset')
    return files

def url_to_name(url):
    name = url.split('/')[-1]
    name = name.split(' ')[0]
    name = name.replace('.tar','')
    return name

def remove_nonalpha(s):
    ns = ''
    for c in s:
        if not c.isdigit():
            ns += c
    
    return ns

def break_audio_chunks(hf_files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    hf_token = os.environ['HF_TOKEN']
    urls = []
    for hf_file in hf_files:
        url = f"https://huggingface.co/datasets/cmeraki/youtube_webdataset/resolve/main/{hf_file}"
        url = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {hf_token}'"
        urls.append(url)
    
    cache_dir = Path('~/.cache/wds/tmp/').expanduser()
    os.makedirs(cache_dir, exist_ok=True)

    transcriber = Transcriber(device='cuda:0')

    cache_size = 10 * (10**9) # GB * 10**9
    dataset = wds.WebDataset(urls, shardshuffle=None, cache_dir=cache_dir, cache_size=cache_size).decode()
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=1
    )

    from tqdm import tqdm
    for elem_id, elem in enumerate(tqdm(dataloader)):
        # try:
        name = url_to_name(elem['__url__']) + f'_{elem_id}'
        # data = elem['json']
        # speaker_id = data['speaker_id']
        speaker_id = remove_nonalpha(name)
        chunk_iterator = transcriber.get_transcript(audio_path=elem['wav'], verbose=False)
        
        for chunk_id, chunk in enumerate(chunk_iterator):
            sample = {}
            audio_tensor = chunk['audio']
            chunk.pop('audio', None)

            sample['raw_text'] = chunk['text']
            sample['speaker_id'] = speaker_id
            sample['metadata'] = chunk
            
            sample_id = f'{name}_chunk_{chunk_id}'
            sample_json_path = output_dir / f'{sample_id}.json'
            sample_wav_path = output_dir / f'{sample_id}.wav'
            
            save_audio(sample_wav_path, audio_tensor)
            
            with open(sample_json_path, 'w') as jsp:
                sample_json = json.dumps(sample, ensure_ascii=False)
                jsp.write(sample_json)
            

        # except:
        #     print('failed')

def whisper_on_hindi():
    is_english = 'en__'
    is_hindi = 'hi__'

    files = get_all_tars_in_hf_repo('cmeraki/youtube_webdataset')
    files = list(filter(lambda x: x.startswith(is_hindi), files))
    print(files)
    break_audio_chunks(files,'~/.cache/chunks/')

whisper_on_hindi()