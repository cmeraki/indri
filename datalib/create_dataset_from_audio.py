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

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
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
def make_transcript(audio_file_path):

    wav = read_audio('path_to_audio_file')
    speech_timestamps = get_speech_timestamps(wav, silero)
    print(speech_timestamps)
    
    transcript = {}
    text_till_now = ""

    for chunk in chunks:
        result = whisper_model.transcribe(audio_file_path, 
                                verbose=True, 
                                word_timestamps=True, 
                                initial_prompt=text_till_now)
        
    


def process(dsname, input_audio_dir, speaker_name):
    dataset = Dataset(repo_id=dsname)
    audio_files = find_audio_files(input_audio_dir)

    print("num audio files", len(audio_files))
    print(audio_files[51])
    audio_files = audio_files[51:52]

    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        make_transcript(audio_file)

        # transcript_path = Path(audio_file).with_suffix('.json')
        # transcript_json = json.dumps(transcript, ensure_ascii=False)

        # with open(transcript_path, 'w') as transcript_writer: 
        #     transcript_writer.write(transcript_json)


        # # sentences = make_sentences(transcript)
        # sentences = transcript["chunks"]
        # audio = AudioSegment.from_file(audio_file)
        
        # for chunk_idx, elem in tqdm(enumerate(sentences), desc='processing chunks..'):
        #     start = elem['timestamp'][0]
        #     end = elem['timestamp'][1]
            
        #     segment = audio[start*1000:end*1000]

        #     id = f'{file_idx}_chunk_{chunk_idx}'
        #     sample = Dataset.create_sample(id=id)
            
        #     sample.raw_text = elem['text']
        #     sample.speaker_id = speaker_name
        #     sample.audio_array = np.asarray(segment.get_array_of_samples())
            
        #     sample.sampling_rate = audio.frame_rate
        #     sample.duration = end - start
        #     sample.metadata = elem
            
        #     audio_path = dataset.get_absolute_path(sample.audio_path)
        #     segment.export(audio_path, format='wav')
        #     # dataset.add_audio(sample)
        #     dataset.add_metadata(sample)
            

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