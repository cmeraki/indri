from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Tuple

class Speakers(Enum):
    SPEAKER_1 = 'Speaker 1'
    SPEAKER_2 = 'Speaker 2'
    SPEAKER_3 = 'Speaker 3'

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
