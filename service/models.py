from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Tuple

class Speakers(Enum):
    SPEAKER_1 = 'Male'
    SPEAKER_2 = 'Female'
    SPEAKER_3 = 'Storyteller'

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[Speakers] = None

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

class TTSSpeakersResponse(BaseModel):
    speakers: List[str]

def speaker_mapping(speaker: Speakers) -> str:
    if speaker == Speakers.SPEAKER_1:
        return '[spkr_hifi_tts_9017]'
    elif speaker == Speakers.SPEAKER_2:
        return '[spkr_hifi_tts_92]'
    elif speaker == Speakers.SPEAKER_3:
        return '[spkr_jenny_jenny]'
    else:
        raise ValueError(f'Speaker {speaker} not supported')
