from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Tuple

class Speakers(Enum):
    SPEAKER_1 = '[spkr_hifi_tts_9017]'
    SPEAKER_2 = '[spkr_jenny_jenny]'
    # English YouTube Speakers
    SPEAKER_3 = '[spkr_youtube_webds_en_akshat]' #b
    SPEAKER_4 = '[spkr_youtube_webds_en_historyofindia]' #g
    SPEAKER_5 = '[spkr_youtube_webds_en_mkbhd]' #g
    SPEAKER_6 = '[spkr_youtube_webds_en_secondhandstories]' #g
    SPEAKER_7 = '[spkr_youtube_webds_en_storiesofmahabharatha]' #g
    SPEAKER_8 = '[spkr_youtube_webds_en_teded]' #chaotic good 

    # Hindi YouTube Speakers
    SPEAKER_9 = '[spkr_youtube_webds_hi_a2motivation]' #g
    SPEAKER_10 = '[spkr_youtube_webds_hi_akshat]' #b
    SPEAKER_11 = '[spkr_youtube_webds_hi_dhruvrathee]' #b
    SPEAKER_12 = '[spkr_youtube_webds_hi_hindiaudiobooks]' #g
    SPEAKER_13 = '[spkr_youtube_webds_hi_kabitaskitchen]' #g
    SPEAKER_14 = '[spkr_youtube_webds_hi_mrbeast]' #b
    SPEAKER_15 = '[spkr_youtube_webds_hi_neelimaaudiobooks]' #g
    SPEAKER_16 = '[spkr_youtube_webds_hi_physicswallah]' #b 
    SPEAKER_17 = '[spkr_youtube_webds_hi_pmmodi]'#m
    SPEAKER_18 = '[spkr_youtube_webds_hi_ranveerallahbadia]' #m
    SPEAKER_19 = '[spkr_youtube_webds_hi_sandeepmaheshwari]' #b
    SPEAKER_20 = '[spkr_youtube_webds_hi_technicalguruji]' #b
    SPEAKER_21 = '[spkr_youtube_webds_hi_unacademyjee]' #b
    SPEAKER_22 = '[spkr_youtube_webds_hi_vivekbindra]' #m
    SPEAKER_23 = '[spkr_youtube_webds_en_derekperkins]' #g
    SPEAKER_24 = '[spkr_youtube_webds_en_mukesh]' #g
    SPEAKER_25 = '[spkr_youtube_webds_en_attenborough]' #g
    SPEAKER_26 = '[spkr_youtube_webds_hi_warikoo]' #g

class TTSRequest(BaseModel):
    text: str
    speaker: Speakers

class TTSMetrics(BaseModel):
    time_to_first_token: List[float]
    time_to_last_token: List[float]
    time_to_decode_audio: float
    input_tokens: List[int]
    decoding_tokens: List[int]
    generate_end_to_end_time: float
    end_to_end_time: Optional[float] = None

class TTSResponse(BaseModel):
    array: str
    dtype: str
    shape: Tuple
    sample_rate: int
    metrics: Optional[TTSMetrics] = None
    request_id: Optional[str] = None

class TTSSpeakersResponse(BaseModel):
    speakers: List[str]

def speaker_mapping(speaker: Speakers) -> str:
    spkr_map = {
        Speakers.SPEAKER_1: '[spkr_hifi_tts_9017]',
        Speakers.SPEAKER_2: '[spkr_jenny_jenny]',
        Speakers.SPEAKER_3: '[spkr_youtube_webds_en_akshat]',
        Speakers.SPEAKER_4: '[spkr_youtube_webds_en_historyofindia]',
        Speakers.SPEAKER_5: '[spkr_youtube_webds_en_mkbhd]',
        Speakers.SPEAKER_6: '[spkr_youtube_webds_en_secondhandstories]',
        Speakers.SPEAKER_7: '[spkr_youtube_webds_en_storiesofmahabharatha]',
        Speakers.SPEAKER_8: '[spkr_youtube_webds_en_teded]',
        Speakers.SPEAKER_9: '[spkr_youtube_webds_hi_a2motivation]',
        Speakers.SPEAKER_10: '[spkr_youtube_webds_hi_akshat]',
        Speakers.SPEAKER_11: '[spkr_youtube_webds_hi_dhruvrathee]',
        Speakers.SPEAKER_12: '[spkr_youtube_webds_hi_hindiaudiobooks]',
        Speakers.SPEAKER_13: '[spkr_youtube_webds_hi_kabitaskitchen]',
        Speakers.SPEAKER_14: '[spkr_youtube_webds_hi_mrbeast]',
        Speakers.SPEAKER_15: '[spkr_youtube_webds_hi_neelimaaudiobooks]',
        Speakers.SPEAKER_16: '[spkr_youtube_webds_hi_physicswallah]',
        Speakers.SPEAKER_17: '[spkr_youtube_webds_hi_pmmodi]',
        Speakers.SPEAKER_18: '[spkr_youtube_webds_hi_ranveerallahbadia]',
        Speakers.SPEAKER_19: '[spkr_youtube_webds_hi_sandeepmaheshwari]',
        Speakers.SPEAKER_20: '[spkr_youtube_webds_hi_technicalguruji]',
        Speakers.SPEAKER_21: '[spkr_youtube_webds_hi_unacademyjee]',
        Speakers.SPEAKER_22: '[spkr_youtube_webds_hi_vivekbindra]',
        Speakers.SPEAKER_23: '[spkr_youtube_webds_en_derekperkins]',
        Speakers.SPEAKER_24: '[spkr_youtube_webds_en_mukesh]',
        Speakers.SPEAKER_25: '[spkr_youtube_webds_en_attenborough]',
        Speakers.SPEAKER_26: '[spkr_youtube_webds_hi_warikoo]'
    }

    speaker_val = spkr_map.get(speaker)

    if speaker_val:
        return speaker_val

    raise ValueError(f'Speaker {speaker} not supported')
