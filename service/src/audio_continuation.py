import sys
sys.path.append('omni/')

import time
import torch
import torchaudio
import numpy as np
import pyloudnorm as pyln
from functools import partial
from typing import List, Optional

from vllm import SamplingParams, RequestOutput
from vllm.inputs import TokensPrompt

from commons import MIMI
from commons import Config as cfg

from .engine import (
    Engine,
    AudioTokenizer,
    TextTokenizer
)
from ..models import AudioOutput
from ..utils import (
    alternative_logits_processor,
    deserialize_tokens
)
from ..logger import get_logger

logger = get_logger(__name__)

class AudioContinuation:
    """Audio continuation"""
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.lm_engine = Engine(model_path, device, gpu_memory_utilization=0.4)
        self.audio_tokenizer = AudioTokenizer(device)
        self.text_tokenizer = TextTokenizer(model_path)

        logits_processor_kwargs = {
            'num_codebooks': cfg.n_codebooks,
            'codebook_size': cfg.per_codebook_size,
            'offset': cfg.OFFSET[MIMI],
            'stop_token': self.text_tokenizer.stop_token
        }
        logits_processors = [
            partial(alternative_logits_processor, **logits_processor_kwargs)
        ]

        self.sampling_params = SamplingParams(
            temperature=0.9,
            top_k=50,
            stop_token_ids=self.text_tokenizer.stop_token,
            max_tokens=1024,
            logits_processors=logits_processors
        )

    async def generate_async(self,
        audio: torch.Tensor,
        sample_rate: int,
        request_id: Optional[str] = None
    ) -> AudioOutput:
        start_time = time.time()
        metrics = {}

        audio = torchaudio.transforms.Resample(sample_rate, 24000)(audio)
        audio = pyln.normalize.peak(audio.numpy(), -1.0)
        audio_tokens, encode_time = self.audio_tokenizer.encode(torch.from_numpy(audio).unsqueeze(0))
        metrics['time_to_encode_audio'] = encode_time

        input_tokens = self.text_tokenizer.prepare_audio_continuation_tokens(audio_tokens)
        logger.info(f'Input tokens shape: {len(input_tokens)}', extra={'request_id': request_id})

        prompt = TokensPrompt(prompt_token_ids=input_tokens)
        results_generator = self.lm_engine.engine.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            request_id=request_id
        )

        preds: List[RequestOutput] = []

        async for request_output in results_generator:
            if request_output.finished:
                preds.append(request_output)

        output_tokens = []
        metrics['time_to_first_token'] = []
        metrics['time_to_last_token'] = []
        metrics['input_tokens'] = []
        metrics['decoding_tokens'] = []

        for idx, request_output in enumerate(preds):
            o = np.array(request_output.outputs[0].token_ids)
            end = np.where(o == self.text_tokenizer.stop_token[0])[0]

            if len(end) > 0:
                end = end[0]
            else:
                end = len(o)

            o = o[:end]
            o = o - cfg.OFFSET[MIMI]
            o = deserialize_tokens(o)
            assert np.all(o >= 0), f'Negative token index generated for batch {idx}'

            metrics['time_to_first_token'].append(
                request_output.metrics.first_token_time - request_output.metrics.first_scheduled_time
            )
            metrics['time_to_last_token'].append(
                request_output.metrics.finished_time - request_output.metrics.first_scheduled_time
            )
            metrics['input_tokens'].append(len(request_output.prompt_token_ids))
            metrics['decoding_tokens'].append(len(request_output.outputs[0].token_ids))

            output_tokens.append(o)

        output_tokens = np.concatenate(output_tokens, axis=1)
        logger.info(f'Output tokens shape: {output_tokens.shape}', extra={'request_id': request_id})

        audio_out, decode_time = self.audio_tokenizer.decode(output_tokens)
        audio_out = pyln.normalize.peak(audio_out, -1.0)

        metrics['time_to_decode_audio'] = decode_time
        metrics['generate_end_to_end_time'] = time.time() - start_time

        return AudioOutput(audio=audio_out, sample_rate=24000, audio_metrics=metrics)

async def main():
    import uuid
    import torchaudio

    audio, sr = torchaudio.load('service/data/modi.sample1.real.wav')
    audio = audio[:, :sr * 5]

    model = AudioContinuation('cmeraki/hf-audio-continue', 'cuda:0')
    result = await model.generate_async(
        audio=audio,
        sample_rate=sr,
        request_id=str(uuid.uuid4())
    )

    torchaudio.save('output.wav', torch.from_numpy(result.audio), sample_rate=result.sample_rate, format='mp3')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
