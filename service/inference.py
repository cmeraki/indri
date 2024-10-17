import sys
sys.path.append('omni/')

import base64
from typing import Tuple, List
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omni.logger import get_logger
from service.tts import TTS

logger = get_logger(__name__)

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

class TTSRequest(BaseModel):
    text: str

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

    results = model.generate(requests.text)
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
        "speakers": ['Speaker 1', 'Speaker 2', 'Speaker 3']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)