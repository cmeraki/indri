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

def load_huggingface_model(device):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from datasets import load_dataset


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=10,
        batch_size=16,  # batch size for inference - set based on your device
        torch_dtype=torch_dtype,
        device=device,
    )

    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]

    # result = pipe(sample)
    # print(result["text"])

    return pipe

def get_transcript(audio_file):
    print(f"whisper:{audio_file}")
    
    # with torch.inference_mode():
    #     result = model.transcribe(audio_file, verbose=True, word_timestamps=True)
    # return result

    audio, sr = torchaudio.load(audio_file)
    print(sr, audio.shape)
    audio = convert_audio(audio, sr=sr, target_sr=16000, target_channels=1)[0]
    results = pipe(audio.numpy(), return_timestamps=True, generate_kwargs={"language": "hindi"})
    return results

# def make_sentences(result):
#     sentences = result['segments']
#     # sent = []
#     # for segment in tqdm(result['segments'], desc='making sentences..'):
#     #     text = segment['text']
#     #     if text:
#     #         sent.append(segment)
#     #         if text[-1] in {".", "。", "!", "！", "?", "？", "|"}:
#     #             start = sent[0]['start']
#     #             end = sent[-1]['end']
#     #             _text = "".join([e['text'] for e in sent])
#     #             sent = {'start': start, 'end': end, 'text': _text}
#     #             sentences.append(sent)
#     #             sent = []
        
#     return sentences



def find_audio_files(folder):
    audio_extensions = ('.mp3', '.flac', '.wav', '.ogg')
    audio_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(audio_extensions):
                audio_files.append(os.path.join(root, file))

    return audio_files


def process(dsname, input_audio_dir, speaker_name):
    dataset = Dataset(repo_id=dsname)
    audio_files = find_audio_files(input_audio_dir)

    print("num audio files", len(audio_files))
    print(audio_files[51])
    audio_files = audio_files[51:52]
    
    for file_idx, audio_file in tqdm(enumerate(audio_files), 'processing file ..'):
        transcript = get_transcript(audio_file)
        transcript_path = Path(audio_file).with_suffix('.json')
        print(transcript_path)
        transcript_json = json.dumps(transcript, ensure_ascii=False)

        with open(transcript_path, 'w') as transcript_writer: 
            transcript_writer.write(transcript_json)


        # sentences = make_sentences(transcript)
        sentences = transcript["chunks"]
        audio = AudioSegment.from_file(audio_file)
        
        for chunk_idx, elem in tqdm(enumerate(sentences), desc='processing chunks..'):
            start = elem['timestamp'][0]
            end = elem['timestamp'][1]
            
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

    pipe = load_huggingface_model(args.device)

    # model = whisper.load_model("turbo", device=args.device)
    # model.eval()
    
    process(args.dsname, args.audio, args.speaker)