import torch
import openai
import numpy as np
from typing import Optional
import pyloudnorm as pyln

from .commons import MIMI
from .commons import Config as cfg

from .engine import (
    AudioTokenizer,
    TextTokenizer
)
from .utils import (
    sanitize_text,
    deserialize_tokens
)
from .logger import get_logger


logger = get_logger(__name__)

class TTS_GGML:
    def __init__(self, model_path: str, device: Optional[str] = 'cpu'):
        self.lm_engine = openai.OpenAI(base_url=model_path)
        self.audio_tokenizer = AudioTokenizer(device)
        self.text_tokenizer = TextTokenizer('11mlabs/indri-0.1-124m-tts')

    def generate(self,
        text: str,
        speaker: str,
    ) -> np.ndarray:

        batch_text = sanitize_text(text)[0]
        input_text = self.text_tokenizer.text_modality_token + batch_text + self.text_tokenizer.convert_token + self.text_tokenizer.acoustic_modality_token + self.text_tokenizer._tokenizer.encode(speaker)

        logger.info(f'Texts after preprocessing: {input_text}')

        response = self.lm_engine.chat.completions.create(
            model='',
            messages=[{'role': 'user', 'content': input_text}],
            temperature=0.5,
            top_k=15,
            max_tokens=1024
        )

        ## this will be a text response, we need to convert it to back to tokens
        preds = response.choices[0].message.content
        preds = preds.split(self.text_tokenizer.stop_token[0])[0]

        logger.info(f'Preds after decoding: {preds}')

        preds = self.text_tokenizer.decode(preds)
        preds -= cfg.OFFSET[MIMI]
        preds = deserialize_tokens(preds, cfg.n_codebooks)
        assert np.all(preds >= 0), f'Negative token index generated'

        mimi_tokens = [preds]

        mimi_tokens = np.concatenate(mimi_tokens, axis=1)
        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        audio_output, decode_time = self.audio_tokenizer.decode(mimi_tokens)
        audio_output = pyln.normalize.peak(audio_output, -1.0)

        return audio_output, 24000

def main():
    from .models import Speakers
    import torchaudio
    
    model = TTS_GGML('http://localhost:8080/completion/')

    # Pure TTS
    audio, sr = model.generate(
        'Everything is much sharper than the very pixelated looking metaglasses. Snapchat had similar glasses which failed miserably.',
        speaker=Speakers.SPEAKER_2,
    )

    torchaudio.save('output_tts.wav', torch.from_numpy(audio), sample_rate=sr, format='mp3')

if __name__ == '__main__':
    main()