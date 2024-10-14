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
from tqdm import tqdm
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg', '.m4a')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    audio_files = audio_files
    return audio_files


def join_chunks(speech_timestamps):
    sampling_rate = 16000
    max_chunk_size = 10*sampling_rate # secs * sr
    max_silence_in_chunk = .5*sampling_rate # secs * sr
    min_chunk_size = 1 * sampling_rate # secs * sr 

    new_timestamps = speech_timestamps[0]

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



class Transcriber:
    def __init__(self, device):
        self.silero = load_silero_vad()
        self.model = whisper.load_model("turbo", device=device)
        self.model.eval()

    @torch.inference_mode()
    def get_transcript(self, audio_path, tmp_dir='/tmp/transcribe/', verbose=True):
        
        Path(tmp_dir).mkdir(exist_ok=True)
        
        # langprobs = self.detect_language(audio_path)
        # print("Language probs", langprobs)

        wav = read_audio(audio_path)
        speech_timestamps = get_speech_timestamps(wav, self.silero, threshold=0.4, min_silence_duration_ms=250, max_speech_duration_s=15)
        speech_timestamps = join_chunks(speech_timestamps)

        # take a large chunk and determine language
        start_of_speech = speech_timestamps[0]['start']
        chunk = wav[start_of_speech:start_of_speech + 16000*30].numpy()
        res = self.model.transcribe(chunk)
        prompt = res['text']
        language = res['language']
        
        print("DETECTED LANGUAGE", res['language']) 
        print("PROMPT", prompt)
        print("NUM CHUNKS", len(speech_timestamps))


        if language not in {'hi', 'en'}:
            language = 'hi'

        updated_prompt = prompt
        for idx, timestamp in enumerate(speech_timestamps):
            subwav = wav[timestamp['start']:timestamp['end']]
                
            transcript = self.model.transcribe(subwav,
                                                initial_prompt=updated_prompt)
            
            updated_prompt = prompt + transcript['text']
    
            if verbose:
                print(timestamp, transcript['text'])
                
            transcript['audio'] = subwav
            yield transcript

    
def process(dsname, input_audio_dir, speaker_name, device):
    dataset = Dataset(repo_id=dsname)
    audio_files = find_audio_files(input_audio_dir)

    print("num audio files", len(audio_files))
    transcriber = Transcriber(device=device)

    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        try:
            print("WORKING ON", audio_file)
            transcript = transcriber.get_transcript(audio_file, verbose=False)
            for chunk_idx, elem in enumerate(transcript):
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

        except:
            print("FAILED", audio_file)    
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--audio', type=str, required=True, help='directory of audio files')
    parser.add_argument('--speaker', type=str, required=False, default=None, help='name of speaker if known')
    parser.add_argument('--dsname', type=str, required=True, help='name of your dataset. will be uploaded to cmeraki/')
    parser.add_argument('--device', type=str, required=True, help='name of device')

    
    args = parser.parse_args()


    
    process(args.dsname, args.audio, args.speaker, args.device)