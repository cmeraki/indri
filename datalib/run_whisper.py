import whisperx
import gc
import json
from tqdm import tqdm


from glob import glob
from pathlib import Path

import sys
# _filter = sys.argv[1]
# _device = int(sys.argv[2])

def download_model():
    from huggingface_hub import snapshot_download
    repo_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
    local_dir = "faster-whisper-large-v3-turbo-ct2"
    snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model")


def get_audio_files(audio_dir):
    audio_files = glob(f'{audio_dir}/*.wav')
    print("Found Audio Files", len(audio_files))
    
    print(len(audio_files))
    return audio_files

def transcribe(audio_file, transcription_path, language, model):
    audio = whisperx.load_audio(audio_file)
    
    result = model.transcribe(audio, batch_size=128, language=language, chunk_size=10)
    transcription = result["segments"] # before alignment
    transcription_json = json.dumps(transcription, ensure_ascii=False)
    with open(transcription_path, 'w') as f:
        f.write(transcription_json)


def transcribe_all(audio_dir, transcription_dir, language, model, device):
    audio_files = get_audio_files(audio_dir)
    transcription_dir = Path(transcription_dir)
    transcription_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model,
                                device='cuda', 
                                device_index=device,
                                compute_type='float16')
    
    for audio_file in tqdm(audio_files):
        audio_file_name = Path(audio_file).stem
        print(audio_file_name)
        transcription_path = transcription_dir / f'{audio_file_name}.json'
        transcribe(audio_file, transcription_path, language, model)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()

    parser.add_argument('--inp', type=str, required=True, help='wav files')
    parser.add_argument('--out', type=str, required=True, help='output transcriptions dir')
    parser.add_argument('--device', type=int, required=True, help='cuda:N')
    parser.add_argument('--language', type=str, required=True, help='en/hi/ka')
    parser.add_argument('--model', type=str, required=True, help='large')

    args = parser.parse_args()

    transcribe_all(args.inp, args.out, args.language, args.model, args.device)

    