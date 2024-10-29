import sys
sys.path.append('omni/')

import time
import base64
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .models import TTSRequest, TTSResponse, TTSSpeakersResponse, Speakers, speaker_mapping
from .logger import get_logger

logger = get_logger(__name__)

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
    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received text: {requests.text} with speaker: {requests.speaker}', extra={'request_id': request_id})

    try:
        results = model.generate(requests.text, speaker_mapping(requests.speaker))
        audio = results['audio']
        metrics = results['metrics']
    except Exception as e:
        logger.critical(f'Error in model generation: {e}', extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

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
        "speakers": [s for s in Speakers]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
