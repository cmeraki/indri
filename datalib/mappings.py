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
    id = "jenny_" + item['file_name'].replace('/', '_')

    # Prepare the JSON data
    json_data = {
        "id": id,
        "raw_text": item['transcription_normalised'],
        "speaker_id": "jenny",
        "sampling_rate": 16000,
        "dataset": "jenny_train",
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


@register(dsname='expresso', hfds='ylacombe/expresso', split='train')
def prepare_expresso(item):
    id = "expresso_" + item['id']
    json_data = {
        "id": id,
        "raw_text": item['text'],
        "speaker_id": item['speaker_id'],
        "sampling_rate": 16000,
        "dataset": "expresso_train",
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


@register(dsname='hifi_tts', hfds='MikhailT/hifi-tts', name='clean', split='train')
def prepare_hifi_tts(item):
    id = "hifi_tts_" + item['file'].replace('/', '_')

    json_data = {
        "id": id,
        "raw_text": item['text_no_preprocessing'],
        "speaker_id": item['speaker'],
        "sampling_rate": 16000,
        "dataset": "hifi_tts_clean_train",
        "metadata": {
            "language": "en",
            "duration": item['duration']
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }
    
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
    id = "ljspeech_" + item['id']

    json_data = {
        "id": id,
        "raw_text": item['text'],
        "speaker_id": "ljspeech",
        "sampling_rate": 16000,
        "dataset": "ljspeech_train",
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


@register(dsname='mls_eng_10k', hfds='parler-tts/mls_eng_10k', split='train')
def prepare_mlseng(item):
    id = "mls_eng_10k_" + item['audio']['path'].replace('/', '_')

    json_data = {
        "id": id,
        "raw_text": item['transcript'],
        "speaker_id": item['speaker_id'],
        "sampling_rate": item['audio']['sampling_rate'],
        "dataset": "mls_eng_10k_train",
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


@register(dsname='gigaspeech', name='xl', split='train', hfds='speechcolab/gigaspeech')
def prepare_gigaspeech(item):
    id = "gs_" + item['segment_id']

    json_data = {
        "id": id,
        "raw_text": item['text'],
        "speaker_id": None,
        "sampling_rate": item['audio']['sampling_rate'],
        "dataset": "gigaspeech_xl_train",
        "metadata": {
            "language": "en"
        }
    }

    wav_data = audio_to_wav_bytestring(item['audio']['array'], item['audio']['sampling_rate'])

    sample = {
        "__key__": id,
        "json": json.dumps(json_data),
        "wav": wav_data
    }

    return sample


# @register(dsname='emilia', split='en', hfds='amphion/Emilia-Dataset')
def prepare_emilia(item):
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