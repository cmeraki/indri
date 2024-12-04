import torch
import openai
import numpy as np
from typing import Optional, List
import pyloudnorm as pyln

from .commons import MIMI, TEXT, CONVERT
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

    def prepare_input_task_str(self, text: str, speaker: str) -> str:

        return cfg.MODALITY_TOKENS[TEXT] + text + cfg.TASK_TOKENS[CONVERT] + \
            cfg.MODALITY_TOKENS[MIMI] + speaker

    def generate(self,
        text: str,
        speaker: str,
    ) -> np.ndarray:

        batch_text = sanitize_text(text)[0]
        input_text = self.prepare_input_task_str(batch_text, speaker)

        logger.info(f'Texts after preprocessing: {input_text}')

        response = self.lm_engine.completions.create(
            model='indri-0.1-124m-tts',
            prompt=input_text,
            temperature=0.5,
            max_tokens=800
        )

        ## this will be a text response, we need to convert it to back to tokens
        preds = response.content
        preds: np.ndarray = self.text_tokenizer._tokenizer.encode(preds, return_tensors='np')[0]

        end_idx = np.where(preds == self.text_tokenizer.stop_token[0])[0]
        if len(end_idx) > 0:
            preds = preds[:end_idx[0]]

        preds -= cfg.OFFSET[MIMI]
        preds = deserialize_tokens(preds, cfg.n_codebooks)
        assert np.all(preds >= 0), f'Negative token index generated'

        mimi_tokens = [preds]

        mimi_tokens = np.concatenate(mimi_tokens, axis=1)
        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        audio_output, decode_time = self.audio_tokenizer.decode(mimi_tokens)
        audio_output = pyln.normalize.peak(audio_output, -1.0)

        return audio_output, 24000

def main(model_path: str, text: str, speaker: str, out_file: str):
    import torchaudio

    model = TTS_GGML(model_path)

    audio, sr = model.generate(text, speaker=speaker)

    torchaudio.save(out_file, torch.from_numpy(audio), sample_rate=sr, format='mp3')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='http://localhost:8080/')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--speaker', type=str, required=True)
    parser.add_argument('--out', type=str, required=False, default='output_tts.wav')

    args = parser.parse_args()

    main(args.model, args.text, args.speaker, args.out)
