import io
import time
import json
import torch
import random
import traceback
import torchaudio
from torio.io import CodecConfig

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from src.tts_gguf import TTS_GGUF
from src.models import (
    TTSSpeakersResponse, Speakers, TTSRequest,
    SpeakerTextRequest, SpeakerTextResponse,
    AudioOutput, TTSMetrics
)
from src.logger import get_logger
from src.models import SPEAKER_MAP
logger = get_logger(__name__)

app = FastAPI()

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    start_time = time.time()
    logger.info(f'Received text: {request.text} with speaker: {request.speaker}')

    try:
        speaker = SPEAKER_MAP.get(request.speaker, {'id': None}).get('id')

        if speaker is None:
            raise HTTPException(status_code=400, detail=f'Speaker {speaker} not supported')

        results: AudioOutput = tts_model.generate(
            text=request.text,
            speaker=speaker
        )
        metrics: TTSMetrics = results.audio_metrics

        audio_tensor = torch.from_numpy(results.audio)
        logger.info(f'Audio shape: {audio_tensor.shape}')

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
        logger.critical(f"Error in model generation: {e}\nStacktrace: {''.join(traceback.format_tb(e.__traceback__))}")
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    metrics.end_to_end_time = end_time - start_time

    logger.info(f'Metrics: {metrics}')

    headers = {
        "Content-Type": "audio/wav",
        "Content-Disposition": "attachment; filename=speech_completion.wav",
        "x-metrics": json.dumps(metrics.model_dump())
    }

    logger.info(f'Metrics: {metrics}')

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


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='http://localhost:8080/', help='Server endpoint for the llama model')
    parser.add_argument('--port', type=int, default=8000, required=False, help='Port to run the server on')

    args =  parser.parse_args()

    logger.info(f'Loading model from {args.model_path} and starting server on port {args.port}')

    global tts_model
    tts_model = TTS_GGUF(model_path=args.model_path)

    server = uvicorn.Server(config=uvicorn.Config(app, host="0.0.0.0", port=args.port))
    server.run()
