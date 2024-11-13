import io
import time
import json
import uuid
import torch
import random
import traceback
import torchaudio
from typing import Optional
from pathlib import Path
from torio.io import CodecConfig

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from .tts import TTS
from .src.audio_continuation import AudioContinuation
from .models import (
    TTSRequest, TTSResponse, TTSSpeakersResponse, Speakers, TTSMetrics,
    SpeakerTextRequest, SpeakerTextResponse, AudioFeedbackRequest,
    AudioOutput
)
from .models import SPEAKER_MAP
from .logger import get_logger
from .launcher import _add_shutdown_handlers
from .db.feedback import RealFakeFeedbackDB

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
    expose_headers=["x-sample-id", "x-request-id", "x-metrics"]
)

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

@app.post("/tts")
async def text_to_speech(text: str, speaker: str, file: Optional[UploadFile] = File(...)):
    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received text: {text} with speaker: {speaker}', extra={'request_id': request_id})

    try:
        speaker = SPEAKER_MAP.get(speaker, {'id': None}).get('id')

        if speaker is None:
            raise HTTPException(status_code=400, detail=f'Speaker {speaker} not supported')

        contents = await file.read()
        if contents is not None:
            logger.info(f'Received audio file', extra={'request_id': request_id})
            audio, sr = torchaudio.load(contents)
        else:
            audio, sr = None, None

        results: AudioOutput = await tts_model.generate_async(
            text=text,
            speaker=speaker,
            audio=audio,
            sample_rate=sr,
            request_id=request_id
        )
        metrics: TTSMetrics = results.audio_metrics

        audio_tensor = torch.from_numpy(results.audio)
        logger.info(f'Audio shape: {audio_tensor.shape}', extra={'request_id': request_id})

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate=results.sample_rate,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
        buffer.seek(0)

    except Exception as e:
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}", extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(request_id) + ' ' + str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech_completion.wav",
        "x-request-id": request_id,
        "x-metrics": json.dumps(metrics.model_dump())
    }

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    return Response(
        content=buffer.getvalue(),
        headers=headers,
        media_type="audio/wav"
    )

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
    try:
        choice = random.choice(sample_audio_files)
        logger.info(f'Serving sample audio: {choice}')

        aud, sr = torchaudio.load(f'service/data/{choice}.wav')

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
            "x-sample-id": choice
        }

        return Response(
            content=buffer.getvalue(),
            headers=headers,
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(f'Error in sampling audio: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audio_feedback")
async def audio_feedback(request: AudioFeedbackRequest):
    try:
        assert request.id in sample_audio_files, f'Sample audio with id {request.id} not found'
        assert request.feedback in [-1, 1], f'Feedback must be -1 or 1'
    except Exception as e:
        logger.error(f'Error in audio feedback: {e}')
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f'Received audio feedback for {request.id}: {request.feedback}')
    RealFakeFeedbackDB().insert_feedback(request.id, request.feedback)

    return Response(status_code=200)

@app.post("/audio_completion_v2")
async def audio_completion_v2(text: str, file: UploadFile = File(...)):
    """
    Receive a request with a wav file and return a completion
    """
    contents = await file.read()
    audio, sr = torchaudio.load(contents)

    request_id = str(uuid.uuid4())

    start_time = time.time()
    logger.info(f'Received audio completion request', extra={'request_id': request_id})

    try:
        # For audio completion, we don't have a speaker, so we use '[spkr_unk]' 
        speaker = '[spkr_unk]'

        results: AudioOutput = await tts_model.generate_async(
            text=text,
            speaker=speaker,
            audio=audio,
            sample_rate=sr,
            request_id=request_id
        )
        metrics: TTSMetrics = results.audio_metrics

        audio_tensor = torch.from_numpy(results.audio)
        logger.info(f'Audio shape: {audio_tensor.shape}', extra={'request_id': request_id})

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate=results.sample_rate,
            format='mp3',
            encoding='PCM_S',
            bits_per_sample=16,
            backend='ffmpeg',
            compression=CodecConfig(bit_rate=64000)
        )
        buffer.seek(0)

    except Exception as e:
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}", extra={'request_id': request_id})
        raise HTTPException(status_code=500, detail=str(request_id) + ' ' + str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech_completion.wav",
        "x-request-id": request_id,
        "x-metrics": json.dumps(metrics.model_dump())
    }

    logger.info(f'Metrics: {metrics}', extra={'request_id': request_id})

    return Response(
        content=buffer.getvalue(),
        headers=headers,
        media_type="audio/wav"
    )

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='cmeraki/mimi_tts_hf', help='HF model repository id')
    parser.add_argument('--device', type=str, default='cuda:0', required=False, help='Device to use for inference')
    parser.add_argument('--port', type=int, default=8000, required=False, help='Port to run the server on')

    args =  parser.parse_args()

    logger.info(f'Loading model from {args.model_path} on {args.device} and starting server on port {args.port}')

    global tts_model
    global ac_model

    tts_model = TTS(model_path=args.model_path, device=args.device)
    ac_model = AudioContinuation(model_path='cmeraki/hf-audio-continue', device=args.device)

    file_names = list(Path('service/data/').resolve().glob('**/*.wav'))
    logger.info(f'Found {len(file_names)} sample audio files')

    global sample_audio_files
    sample_audio_files = [f.stem for f in file_names]

    server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=args.port))
    _add_shutdown_handlers(app, server)

    server.run()
