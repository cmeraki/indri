import whisper
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
from datalib.datalib import Dataset
import glob
import numpy as np
import torch
import os
import torchaudio

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio
silero = load_silero_vad()

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files

@torch.inference_mode()
def get_transcript(audio_path):
    print("WORKING ON", audio_path)
    wav = read_audio(audio_path)
    speech_timestamps = get_speech_timestamps(wav, silero)
    from tqdm import tqdm
    text = ''
    data = []
    for timestamp in speech_timestamps:
        subwav = wav[timestamp['start']:timestamp['end']]
        transcript = whisper_model.transcribe(audio=subwav, initial_prompt=text) 
        text += transcript['text']
        transcript['audio'] = subwav
        yield transcript
    
    
def process(dsname, input_audio_dir, speaker_name):
    dataset = Dataset(repo_id=dsname)
    audio_files = find_audio_files(input_audio_dir)

    print("num audio files", len(audio_files))
    # print(audio_files[51])
    # audio_files = audio_files[51:52]

    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        transcript = get_transcript(audio_file)

        # transcript_path = Path(audio_file).with_suffix('.json')
        # transcript_json = json.dumps(transcript, ensure_ascii=False)

        # with open(transcript_path, 'w') as transcript_writer: 
        #     transcript_writer.write(transcript_json)
        
        
        for chunk_idx, elem in tqdm(enumerate(transcript), desc='processing chunks..'):
            id = f'{file_idx}_chunk_{chunk_idx}'
            sample = Dataset.create_sample(id=id)
            
            sample.raw_text = elem['text']
            sample.speaker_id = speaker_name
            sample.audio_array = elem['audio'].numpy()
            
            sample.sampling_rate = 16000
            sample.duration = len(sample.audio_array)/sample.sampling_rate
            
            audio_path = dataset.get_absolute_path(sample.audio_path)
            save_audio(audio_path, elem['audio'], sampling_rate=16000)
            # dataset.add_audio(sample)
            dataset.add_metadata(sample)
            

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--audio', type=str, required=True, help='directory of audio files')
    parser.add_argument('--speaker', type=str, required=False, default=None, help='name of speaker if known')
    parser.add_argument('--dsname', type=str, required=True, help='name of your dataset. will be uploaded to cmeraki/')
    parser.add_argument('--device', type=str, required=True, help='name of device')

    
    args = parser.parse_args()

    whisper_model = whisper.load_model("turbo", device=args.device)
    whisper_model.eval()
    
    process(args.dsname, args.audio, args.speaker)