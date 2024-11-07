import io
import time
import base64
import uuid
import random
import traceback
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict
from torio.io import CodecConfig

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .models import (
    TTSRequest, TTSResponse, TTSSpeakersResponse, Speakers, TTSMetrics,
    SpeakerTextRequest, SpeakerTextResponse, AudioFeedbackRequest
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


@app.get("/sample_audio")
async def sample_audio():
    choice = random.choice(list(sample_audio_files.keys()))
    logger.info(f'Serving sample audio: {choice}')

    aud, sr = sample_audio_files[choice]

    buffer = io.BytesIO()
    torchaudio.save(
        buffer,
        aud,
        sample_rate=sr,
        format='mp3',
        encoding='PCM_S',
        bits_per_sample=16,
        backend='ffmpeg',
        compression=CodecConfig(bit_rate=64000)
    )
    buffer.seek(0)

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech.wav",
        "X-Sample-ID": choice
    }

    return Response(
        content=buffer.getvalue(),
        headers=headers,
        media_type="audio/wav"
    )

@app.post("/audio_feedback")
async def audio_feedback(request: AudioFeedbackRequest):
    assert request.id in sample_audio_files, f'Sample audio with id {request.id} not found'
    assert request.feedback in [-1, 1], f'Feedback must be -1 or 1'

    logger.info(f'Received audio feedback for {request.id}: {request.feedback}')
    audio_feedback_counter[request.id] = request.feedback

    return Response(status_code=200)


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

    global sample_audio_files
    sample_audio_files: Dict[str, np.ndarray] = {}

    file_names = list(Path('service/data/').resolve().glob('**/*.wav'))

    logger.info(f'Found {len(file_names)} sample audio files')
    for f in file_names:
        aud, sr = torchaudio.load(f)
        sample_audio_files[f.stem] = (aud, sr)

    global audio_feedback_counter
    audio_feedback_counter: Dict[str, int] = {} # id -> feedback

    server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=args.port))
    _add_shutdown_handlers(app, server)

    server.run()
