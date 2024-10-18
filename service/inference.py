import sys
sys.path.append('omni/')

import time
import base64
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .models import TTSRequest, TTSResponse, TTSSpeakersResponse, Speakers
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
        "https://indrivoice.ai",
        "https://www.indrivoice.ai",
        "https://indrivoice.io",
        "https://www.indrivoice.io",
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

@app.post("/tts", response_model=TTSResponse)
def text_to_speech(requests: TTSRequest):
    start_time = time.time()
    logger.info(f'Received text: {requests.text}')

    try:
        results = model.generate(requests.text)
        audio = results['audio']
        metrics = results['metrics']
    except Exception as e:
        logger.critical(f'Error in model generation: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}')

    encoded = base64.b64encode(audio.tobytes()).decode('utf-8')
    return {
        "array": encoded,
        "dtype": str(audio.dtype),
        "shape": audio.shape,
        "sample_rate": 24000,
        "metrics": metrics
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