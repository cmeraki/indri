import time
import base64
import uuid
import random
import traceback
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .models import (
    TTSRequest, TTSResponse, TTSSpeakersResponse, Speakers, TTSMetrics,
    SpeakerTextRequest, SpeakerTextResponse
)
from .models import SPEAKER_MAP
from .logger import get_logger
from .launcher import _add_shutdown_handlers

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

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(requests: TTSRequest):
    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received text: {requests.text} with speaker: {requests.speaker}', extra={'request_id': request_id})

    try:
        speaker = SPEAKER_MAP.get(requests.speaker, {'id': None}).get('id')

        if speaker is None:
            raise HTTPException(status_code=400, detail=f'Speaker {requests.speaker} not supported')

        results = await model.generate_async(
            requests.text,
            speaker,
            request_id=request_id
        )
        audio: np.ndarray = results['audio']
        metrics: TTSMetrics = results['metrics']
    except Exception as e:
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}", extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(request_id) + ' ' + str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    encoded = base64.b64encode(audio.tobytes()).decode('utf-8')
    return {
        "array": encoded,
        "dtype": str(audio.dtype),
        "shape": audio.shape,
        "sample_rate": 24000,
        "metrics": metrics,
        "request_id": request_id
    }

@app.get("/speakers", response_model=TTSSpeakersResponse)
async def available_speakers():
    return {
        "speakers": [s for s in Speakers]
    }

@app.post("/speaker_text", response_model=SpeakerTextResponse)
async def speaker_text(request: SpeakerTextRequest):
    speaker_text = SPEAKER_MAP.get(request.speaker, {'text': None}).get('text')

    if speaker_text is None:
        raise HTTPException(status_code=400, detail=f'Speaker {request.speaker} not supported')

    return {
        "speaker_text": random.choice(speaker_text)
    }

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='cmeraki/mimi_tts_hf', choices=['cmeraki/mimi_tts_hf', 'cmeraki/mimi_tts_hf_stage'], help='HF model repository id')
    parser.add_argument('--device', type=str, default='cuda:0', required=False, help='Device to use for inference')
    parser.add_argument('--port', type=int, default=8000, required=False, help='Port to run the server on')

    args =  parser.parse_args()

    logger.info(f'Loading model from {args.model_path} on {args.device} and starting server on port {args.port}')

    global model
    model = TTS(
        model_path=args.model_path,
        device=args.device
    )

    server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=args.port))
    _add_shutdown_handlers(app, server)

    server.run()
