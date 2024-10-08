import os
import json
import wave
import numpy as np
import torchaudio
import io

def audio_to_wav_bytestring(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    buffer = io.BytesIO()
    torchaudio.save(buffer,
                   waveform,
                   sample_rate=16000,format='mp3',
                   encoding='PCM_S',
                   bits_per_sample=16,
                   backend='ffmpeg',
                   compression=torchaudio.io.CodecConfig(bit_rate=64000))
    wav_bytestring = buffer.getvalue()
    
    return wav_bytestring, sample_rate


def prepare_local_dataset(dsname, channel_name, audio_name, folder_path, split, language):
    channel_path = os.path.join(folder_path,channel_name)
    audio_path = os.path.join(channel_path,"audio_files_compressed",audio_name) + ".wav"
    subtitle_path = os.path.join(channel_path,"subtitles",audio_name) + ".vtt"
    file_name = os.path.basename(audio_path)
    id = file_name.replace('.', '_')

    if os.path.exists(subtitle_path):
        with open(subtitle_path, 'rb') as f:
            transcription = f.read().strip()
    else:
        print(f"Subtitle not found: {subtitle_path}")
        transcription = None
    
    # Load the audio file using torchaudio
    wav_bytes, sample_rate = audio_to_wav_bytestring(audio_path)

    # Prepare the JSON data
    json_data = {
        "id": id,
        "speaker_id": "Null",  # You might want to adjust this
        "sampling_rate": sample_rate,
        "dataset": "local_dataset_train",
        "metadata": {
            "language": language  # Adjust as needed
        }
    }

    wav_data = audio_to_wav_bytestring(audio_path)

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_bytes,
        "vtt": transcription
    }

    return sample
