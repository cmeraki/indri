from datalib.datalib import Dataset

dataset_info = {} 

def register(dsname, hfds, split=None, name=None):
    def decorator(func):
        dataset_info[dsname] = {
            'hfds' : hfds,
            'method':func,
            'dsname': dsname,  
            'split': split,
            'name': name,
            'path': hfds}
        return func
    return decorator


@register(dsname='jenny', hfds='reach-vb/jenny_tts_dataset')
def prepare_jenny(item):
    audio_format = '.wav'
    id = item['file_name'].replace('/', '_')
    sample = Dataset.create_sample(id, audio_format)
    sample.raw_text = item['transcription_normalised']
    sample.speaker_id = 'jenny'
    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample


@register(dsname='expresso', hfds='ylacombe/expresso')
def prepare_expresso(item):
    audio_format = '.wav'
    sample.id=item['id']
    sample = Dataset.create_sample(id, audio_format)
    
    sample.raw_text = item['text']
    sample.speaker_id = item['speaker_id']

    audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    sample.audio_array = audio_array
    return sample


@register(dsname='hifi_tts', hfds='MikhailT/hifi-tts')
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


@register(dsname='vctk', hfds='sanchit-gandhi/vctk')
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

@register(dsname='globe', hfds='MushanW/GLOBE')
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

@register(dsname='ljspeech', hfds='keithito/lj_speech')
def prepare_ljspeech(item):
    audio_format = '.wav'
    id=item['id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text']
    sample.metadata = {'normalized_text': item['normalized_text']}
    sample.speaker_id = 'ljspeech'

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample


@register(dsname='mls_eng_10k', hfds='parler-tts/mls_eng_10k')
def prepare_mlseng(item):
    audio_format = '.wav'
    id=item['audio']['path']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['transcript']
    sample.speaker_id = item['speaker_id']

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

@register(dsname='gigaspeech', name='xl', split='train', hfds='speechcolab/gigaspeech')
def prepare_gigaspeech(item):
    audio_format = '.wav'
    id=item['segment_id']
    
    sample = Dataset.create_sample(id=id, audio_format=audio_format)
    sample.raw_text = item['text']
    sample.speaker_id = None

    sample.audio_array = item['audio']['array']
    sample.sampling_rate = item['audio']['sampling_rate']
    return sample

# {'__key__': 'EN_B00000_S00000_W000000', 
#  '__url__': 'hf://datasets/amphion/Emilia-Dataset@bcaad00d13e7c101485990a46e88f5884ffed3fc/EN/EN-B000000.tar', 
#  'json': {'dnsmos': 3.2927, 
#           'duration': 6.264, 
#           'id': 'EN_B00000_S00000_W000000', 
#           'language': 'en', 
#           'speaker': 'EN_B00000_S00000', 
#           'text': " You can help my mother and you- No. You didn't leave a bad situation back home to get caught up in another one here. What happened to you, Los Angeles?",
#           'wav': 'EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3'
#           },
          
# 'mp3': {'path': 'EN_B00000_S00000_W000000.mp3',
#         'array': array([-0.00156609, -0.00178264, -0.00098045, ..., -0.00193841, -0.00217433, -0.00080763]),
#         'sampling_rate': 24000}
# }

@register(dsname='emilia', split='en', hfds='amphion/Emilia-Dataset')
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


print(dataset_info)