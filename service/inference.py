import sys
sys.path.append('omni/')

import base64
import numpy as np
from typing import List, Tuple
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, HTTPException

from omni.logger import get_logger
from service.tts import TTS

logger = get_logger(__name__)

app = FastAPI()

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)