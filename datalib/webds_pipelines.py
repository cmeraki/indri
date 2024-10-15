from create_dataset_from_audio import BatchTranscriber
from torch.utils.data import DataLoader
import webdataset as wds
from pathlib import Path
import os
import torchaudio
from huggingface_hub import list_repo_files
from silero_vad import save_audio, load_silero_vad, read_audio, get_speech_timestamps
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

def get_whisper(device):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def join_chunks(speech_timestamps):
    sampling_rate = 16000
    max_chunk_size = 10*sampling_rate # secs * sr
    max_silence_in_chunk = .5*sampling_rate # secs * sr
    min_chunk_size = 1 * sampling_rate # secs * sr 

    new_timestamps = [speech_timestamps[0]]

    for timestamp in speech_timestamps[1:]:
        prev_chunk = new_timestamps[-1]

        gap = prev_chunk['end'] - timestamp['start']
        current_chunk_duration = prev_chunk['end'] - prev_chunk['start']
        prev_chunk_duration = prev_chunk['end'] - prev_chunk['start']

        if prev_chunk_duration < min_chunk_size:
            prev_chunk['end'] = timestamp['end']
         
        elif (current_chunk_duration < max_chunk_size) and (gap < max_silence_in_chunk):
            prev_chunk['end'] = timestamp['end']
        
        else:
            new_timestamps.append(timestamp)
    
    return new_timestamps

def break_audio_chunks(hf_files, output_dir, device):
    silero = load_silero_vad()    
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

    whisper_pipe = get_whisper(device)

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
        name = url_to_name(elem['__url__']) + f'_{elem_id}'
        print("PROCESSING", name)
        
        speaker_id = remove_nonalpha(name)
        wav = read_audio(elem['wav'])
        speech_timestamps = get_speech_timestamps(wav, 
                                                  silero,
                                                  threshold=0.4, 
                                                  min_silence_duration_ms=250, 
                                                  max_speech_duration_s=15)
        
        speech_timestamps = join_chunks(speech_timestamps)
        print("NUM CHUNKS", len(speech_timestamps))
        
        chunk_files = []

        for chunk_id, timestamp in enumerate(speech_timestamps):
            subwav = wav[timestamp['start']:timestamp['end']]
            sample_id = f'{name}_chunk_{chunk_id}'
            sample_wav_path = str(output_dir / f'{sample_id}.wav')
            save_audio(sample_wav_path, subwav)
            chunk_files.append(sample_wav_path)
        
        print(chunk_files)

        generate_kwargs = {
            "max_new_tokens": 256,
            "num_beams": 1,
            "condition_on_prev_tokens": True,
            "compression_ratio_threshold": 2.4,  # zlib compression ratio threshold (in token space)
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }

        transcripts = whisper_pipe(chunk_files,
                                   batch_size=128,
                                   chunk_length_s=20, 
                                   generate_kwargs=generate_kwargs)

        for chunk_id, timestamp in enumerate(speech_timestamps):
            subwav = wav[timestamp['start']:timestamp['end']]
            sample_id = f'{name}_chunk_{chunk_id}'
            sample_json_path = output_dir / f'{sample_id}.json'
            transcript = transcripts[chunk_id]
            print(transcript)
            
            sample = {}
            sample['raw_text'] = transcript['text']
            sample['speaker_id'] = speaker_id
            
            with open(sample_json_path, 'w') as jsp:
                sample_json = json.dumps(sample, ensure_ascii=False)
                jsp.write(sample_json)
            

def run_whisper(lang, outdir, device):
    files = get_all_tars_in_hf_repo('cmeraki/youtube_webdataset')
    files = list(filter(lambda x: x.startswith(lang), files))
    # files = files[6:]
    print(files)
    break_audio_chunks(files, outdir, device)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--lang', type=str, required=True, help='language / or any string filter')
    parser.add_argument('--outdir', type=str, required=True, help='output dir for chunks')
    parser.add_argument('--device', type=str, required=True, help='cuda:0..')
    
    args = parser.parse_args()
    
    run_whisper(args.lang, args.outdir, args.device)