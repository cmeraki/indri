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
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files


class Transcriber:
    def __init__(self, device):
        self.silero = load_silero_vad()

        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    def detect_language(self, audio):
        mel = self.whisper_model.log_mel_spectrogram(audio).to(self.whisper_model.device)
        _, probs = self.whisper_model.detect_language(mel)
        print(_, probs)
        return probs

    @torch.inference_mode()
    def get_transcript(self, audio_path, verbose=True, batch_size=4):
        langprobs = self.detect_language(audio_path)
        print("Language probs", langprobs)

        wav = read_audio(audio_path)
        speech_timestamps = get_speech_timestamps(wav, self.silero)
        
        prev_text = ''
        audio_batch = []

        for timestamp in speech_timestamps:
            subwav = wav[timestamp['start']:timestamp['end']]
            audio_batch.append(subwav)
            if len(audio_batch) >= batch_size:
                transcript = self.pipe(audio=audio_batch,
                                       batch_size=batch_size)
                print(transcript)

                if verbose:
                    print(timestamp, transcript['text'])
                
                text = transcript['text']
            transcript['audio'] = subwav
            yield transcript

    
def process(dsname, input_audio_dir, speaker_name):
    dataset = Dataset(repo_id=dsname)
    audio_files = find_audio_files(input_audio_dir)

    print("num audio files", len(audio_files))
    transcriber = Transcriber()

    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        print("WORKING ON", audio_path)
        try:
            transcript = transcriber.get_transcript(audio_file)
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
        
        except:
            print("FAILED", audio_path)        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--audio', type=str, required=True, help='directory of audio files')
    parser.add_argument('--speaker', type=str, required=False, default=None, help='name of speaker if known')
    parser.add_argument('--dsname', type=str, required=True, help='name of your dataset. will be uploaded to cmeraki/')
    parser.add_argument('--device', type=str, required=True, help='name of device')

    
    args = parser.parse_args()


    
    process(args.dsname, args.audio, args.speaker)