import json
import numpy as np
from datalib.datalib import Dataset
from tts.utils import audio_to_wav_bytestring

dataset_info = {}

def register(dsname, hfds, split=None, name=None):
    def decorator(func):
        dataset_info[dsname] = {
            'hfds'  : hfds,
            'method': func,
            'dsname': dsname,
            'split': split,
            'name': name,
            'path': hfds
        }
        return func
    return decorator


@register(dsname='jenny', hfds='reach-vb/jenny_tts_dataset', split='train')
def prepare_jenny(item):
    id = item['file_name'].replace('/', '_')

    # Prepare the JSON data
    json_data = {
        "id": id,
        "raw_text": item['transcription_normalised'],
        "speaker_id": "jenny",
        "sampling_rate": item['audio']['sampling_rate'],
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data,
        "wav.npy": item['audio']['array'].astype(np.float32)
    }

    return sample


@register(dsname='expresso', hfds='ylacombe/expresso')
def prepare_expresso(item):
    json_data = {
        "id": item['id'],
        "raw_text": item['text'],
        "speaker_id": item['speaker_id'],
        "sampling_rate": item['audio']['sampling_rate'],
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": item['id'],
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


# @register(dsname='hifi_tts', hfds='MikhailT/hifi-tts')
def prepare_hifi_tts(item):
    audio_format = '.wav'
    id=item['file'].replace('/', '_')
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text_no_preprocessing']
    sample.speaker_id = item['speaker']
    sample.duration = item['duration']
    
    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample


# @register(dsname='vctk', hfds='sanchit-gandhi/vctk')
def prepare_vctk(item):
    audio_format = '.wav'
    id = item['text_id'] + '_' + item['speaker_id']
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text']
    sample.speaker_id = item['speaker_id']
    sample.metadata = {k for k in item if k != 'audio'}

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

# @register(dsname='globe', hfds='MushanW/GLOBE')
def prepare_globe(item):
    audio_format = '.wav'
    id = item['text_id'] + '_' + item['speaker_id']
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['transcript']
    sample.speaker_id = item['speaker_id']
    sample.metadata = {k for k in item if k != 'audio'}

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

@register(dsname='ljspeech', hfds='keithito/lj_speech', split='train')
def prepare_ljspeech(item):
    json_data = {
        "id": item['id'],
        "raw_text": item['text'],
        "speaker_id": "ljspeech",
        "sampling_rate": item['audio']['sampling_rate'],
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": item['id'],
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


# @register(dsname='mls_eng_10k', hfds='parler-tts/mls_eng_10k')
def prepare_mlseng(item):
    audio_format = '.wav'
    id=item['audio']['path']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['transcript']
    sample.speaker_id = item['speaker_id']

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

# @register(dsname='gigaspeech', name='xl', split='train', hfds='speechcolab/gigaspeech')
def prepare_gigaspeech(item):
    audio_format = '.wav'
    id=item['segment_id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text']
    sample.speaker_id = None

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

# @register(dsname='emilia', split='en', hfds='amphion/Emilia-Dataset')
def prepare_gigaspeech(item):
    audio_format = '.wav'
    id=item['json']['id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['json']['text']
    sample.speaker_id = item['json']['speaker']

    sample.audio_array = item['mp3']['array']
    sample.sampling_rate = item['mp3']['sampling_rate']
    sample.duration = item['json']['duration']
    sample.metadata = item['json']
    return sample


@register(dsname='libritts', hfds='parler-tts/libritts_r_filtered', name='clean', split='train.clean.360')
def prepare_libritts(item):
    audio_format = '.wav'
    id=item['id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text_original']
    sample.speaker_id = item['speaker_id']

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

@register(dsname='libritts_100', hfds='parler-tts/libritts_r_filtered', name='clean', split='train.clean.100')
def prepare_libritts_100(item):
    audio_format = '.wav'
    id=item['id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text_original']
    sample.speaker_id = item['speaker_id']

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample


@register(dsname='libritts_other', hfds='parler-tts/libritts_r_filtered', name='other', split='train.other.500')
def prepare_libritts_other(item):
    audio_format = '.wav'
    id=item['id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text_original']
    sample.speaker_id = item['speaker_id']

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample


@register(dsname='shrutilipi', hfds='collabora/ai4bharat-shrutilipi', split='train')
def prepare_libritts_other(item):
    audio_format = '.wav'
    id=item['audio']['path'].replace('.wav', '')
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['transcription']
    sample.speaker_id = None

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

print(dataset_info)