from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Tuple

class Speakers(Enum):
    SPEAKER_1 = '[spkr_youtube_webds_en_historyofindia]'
    SPEAKER_2 = '[spkr_youtube_webds_en_mkbhd]'
    SPEAKER_3 = '[spkr_youtube_webds_en_secondhandstories]'
    SPEAKER_4 = '[spkr_youtube_webds_en_storiesofmahabharatha]'
    SPEAKER_5 = '[spkr_youtube_webds_hi_a2motivation]'
    SPEAKER_6 = '[spkr_youtube_webds_hi_hindiaudiobooks]'
    SPEAKER_7 = '[spkr_youtube_webds_hi_kabitaskitchen]'
    SPEAKER_8 = '[spkr_youtube_webds_hi_neelimaaudiobooks]'
    SPEAKER_9 = '[spkr_youtube_webds_en_derekperkins]'
    SPEAKER_10 = '[spkr_youtube_webds_en_mukesh]'
    SPEAKER_11 = '[spkr_youtube_webds_en_attenborough]'
    SPEAKER_12 = '[spkr_youtube_webds_hi_warikoo]'
    SPEAKER_13 = '[spkr_youtube_webds_hi_pmmodi]'


class TTSRequest(BaseModel):
    text: str
    speaker: Speakers

class TTSMetrics(BaseModel):
    time_to_first_token: float
    time_to_last_token: float
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
        Speakers.SPEAKER_1: '[spkr_youtube_webds_en_historyofindia]',
        Speakers.SPEAKER_2: '[spkr_youtube_webds_en_mkbhd]',
        Speakers.SPEAKER_3: '[spkr_youtube_webds_en_secondhandstories]',
        Speakers.SPEAKER_4: '[spkr_youtube_webds_en_storiesofmahabharatha]',
        Speakers.SPEAKER_5: '[spkr_youtube_webds_hi_a2motivation]',
        Speakers.SPEAKER_6: '[spkr_youtube_webds_hi_hindiaudiobooks]',
        Speakers.SPEAKER_7: '[spkr_youtube_webds_hi_kabitaskitchen]',
        Speakers.SPEAKER_8: '[spkr_youtube_webds_hi_neelimaaudiobooks]',
        Speakers.SPEAKER_9: '[spkr_youtube_webds_en_derekperkins]',
        Speakers.SPEAKER_10: '[spkr_youtube_webds_en_mukesh]',
        Speakers.SPEAKER_11: '[spkr_youtube_webds_en_attenborough]',
        Speakers.SPEAKER_12: '[spkr_youtube_webds_hi_warikoo]',
        Speakers.SPEAKER_13: '[spkr_youtube_webds_hi_pmmodi]'
    }

    speaker_val = spkr_map.get(speaker)

    if speaker_val:
        return speaker_val

    raise ValueError(f'Speaker {speaker} not supported')
