import sys
sys.path.append('omni/')

import base64
from enum import Enum
from typing import Tuple, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .logger import get_logger

logger = get_logger(__name__)

# TODO: Propogate speaker to tts
# DONE: Exception handling

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://indri-ui.vercel.app",
        "https://indrivoice.io",
        "https://indri-ui-11mlabs-11mlabs-projects.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

global model
model = TTS(
    'cmeraki/mimi_tts_hf',
    device='cuda:0'
)

class Speakers(Enum):
    SPEAKER_1 = 'Speaker 1'
    SPEAKER_2 = 'Speaker 2'
    SPEAKER_3 = 'Speaker 3'

class TTSRequest(BaseModel):
    text: str
    speaker: Optional[Speakers] = None

class TTSResponse(BaseModel):
    array: str
    dtype: str
    shape: Tuple
    sample_rate: int

class TTSSpeakersResponse(BaseModel):
    speakers: List[str]

@app.post("/tts", response_model=TTSResponse)
def text_to_speech(requests: TTSRequest):
    logger.info(f'Received text: {requests.text}')

    try:
        results = model.generate(requests.text)
    except Exception as e:
        logger.critical(f'Error in model generation: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    encoded = base64.b64encode(results.tobytes()).decode('utf-8')
    return {
        "array": encoded,
        "dtype": str(results.dtype),
        "shape": results.shape,
        "sample_rate": 24000
    }

@app.get("/speakers", response_model=TTSSpeakersResponse)
def available_speakers():
    return {
        "speakers": [
            Speakers.SPEAKER_1,
            Speakers.SPEAKER_2,
            Speakers.SPEAKER_3
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)