import time
import torch
import openai
import numpy as np
from typing import Optional
import pyloudnorm as pyln

from .commons import MIMI, TEXT, CONVERT
from .commons import Config as cfg

from .engine import (
    AudioTokenizer,
    TextTokenizer
)
from .models import AudioOutput, TTSMetrics
from .utils import (
    sanitize_text,
    deserialize_tokens
)
from .logger import get_logger


logger = get_logger(__name__)

class TTS_GGUF:
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
    ) -> AudioOutput:
        start_time = time.time()

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

        mimi_tokens = preds - cfg.OFFSET[MIMI]
        mimi_tokens = deserialize_tokens(mimi_tokens, cfg.n_codebooks)
        assert np.all(mimi_tokens >= 0), f'Negative token index generated'

        mimi_tokens = [mimi_tokens]

        mimi_tokens = np.concatenate(mimi_tokens, axis=1)
        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        audio_output, decode_time = self.audio_tokenizer.decode(mimi_tokens)
        audio_output = pyln.normalize.peak(audio_output, -1.0)

        metrics = TTSMetrics(
            time_to_first_token=[response.timings['prompt_ms']/1000],
            time_to_last_token=[response.timings['predicted_ms']/1000],
            time_to_decode_audio=decode_time,
            input_tokens=[response.tokens_evaluated],
            decoding_tokens=[len(preds)],
            generate_end_to_end_time=time.time()-start_time
        )

        return AudioOutput(audio=audio_output, sample_rate=24000, audio_metrics=metrics)

def main(model_path: str, text: str, speaker: str, out_file: str):
    import torchaudio

    model = TTS_GGUF(model_path)

    result = model.generate(text, speaker=speaker)
    print(result.audio_metrics)

    torchaudio.save(out_file, torch.from_numpy(result.audio), sample_rate=result.sample_rate, format='mp3')

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='http://localhost:8080/')
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--speaker', type=str, required=True)
    parser.add_argument('--out', type=str, required=False, default='output_tts.wav')

    args = parser.parse_args()

    main(args.model, args.text, args.speaker, args.out)
