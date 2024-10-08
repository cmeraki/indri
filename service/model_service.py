import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from omni.logger import get_logger
from omni.train import get_text_tokenizer, TaskGenerator

from service.tts import TTS
import service.utils as utils

logger = get_logger(__name__)

app = FastAPI()

text_tokenizer = get_text_tokenizer()
dl = TaskGenerator(loader=None, full_batches=False)

global model
model = TTS('./tempmodel/')

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio: List[float]

def preprocess_text(text: str) -> List[str]:
    text = infer_utils.normalize_text(text)
    text = infer_utils.split_and_join_sentences(text)

    return text

@app.post("/tts")#, response_model=TTSResponse)
def text_to_speech(requests: TTSRequest):

    texts = preprocess_text(requests.text)
    logger.info(f'Texts after preprocessing: {texts}')

    results = []
    batch_size = 32

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        

    results = [item for sublist in results for item in sublist]
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)