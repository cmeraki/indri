import whisper
import json
from pydub import AudioSegment
from pathlib import Path
from tqdm import tqdm
from datalib.datalib import Dataset
import glob
import numpy as np
import torch

def get_transcript(audio_file):
    print(f"whisper:{audio_file}")
    with torch.inference_mode():
        result = model.transcribe(audio_file, verbose=False, language="en")
    return result


def make_sentences(result):
    sentences = []
    sent = []
    for segment in tqdm(result['segments'], desc='making sentences..'): 
        text = segment['text']
        if text:
            sent.append(segment)
            if text[-1] in {".", "。", "!", "！", "?", "？"}:
                start = sent[0]['start']
                end = sent[-1]['end']
                _text = "".join([e['text'] for e in sent])
                sent = {'start': start, 'end': end, 'text': _text}
                sentences.append(sent)
                sent = []
        
    return sentences

def process(dsname, input_audio_dir, speaker_name):
    dataset = Dataset(repo_id=dsname)
    audio_files = list(glob.glob(input_audio_dir + '*.wav'))
    audio_files += list(glob.glob(input_audio_dir + '*.mp3'))

    print("num audio files", len(audio_files))

    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        transcript = get_transcript(audio_file)
        sentences = make_sentences(transcript)
        audio = AudioSegment.from_file(audio_file)
        
        for chunk_idx, elem in tqdm(enumerate(sentences), desc='processing chunks..'):
            start = elem['start']
            end = elem['end']
            segment = audio[start*1000:end*1000]

            id = f'{file_idx}_chunk_{chunk_idx}'
            sample = Dataset.create_sample(id=id)
            
            sample.raw_text = elem['text']
            sample.speaker_id = speaker_name
            sample.audio_array = np.asarray(segment.get_array_of_samples())
            
            sample.sampling_rate = audio.frame_rate
            sample.duration = end - start
            sample.metadata = elem
            
            audio_path = dataset.get_absolute_path(sample.audio_path)
            segment.export(audio_path, format='wav')
            dataset.add_metadata(sample)
            


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add a hf dataset in 3 steps')
    parser.add_argument('--audio', type=str, required=True, help='Chose one of 3 modes')
    parser.add_argument('--speaker', type=str, required=False, default=None, help='name of speaker if known')
    parser.add_argument('--dsname', type=str, required=True, help='name of your dataset. will be uploaded to cmeraki/')
    parser.add_argument('--device', type=str, required=True, help='name of device')

    
    args = parser.parse_args()
    model = whisper.load_model("turbo", device=args.device)
    model.eval()
    
    process(args.dsname, args.audio, args.speaker)