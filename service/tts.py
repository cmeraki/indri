import sys
sys.path.append('omni/')

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from functools import partial
import pyloudnorm as pyln

from vllm import SamplingParams, RequestOutput
from vllm.inputs import TokensPrompt

from commons import MIMI
from commons import Config as cfg

from .src import (
    VLLMEngine,
    AudioTokenizer,
    TextTokenizer
)
from .utils import (
    alternative_logits_processor,
    sanitize_text,
    deserialize_tokens
)
from .logger import get_logger
from .models import TTSMetrics

logger = get_logger(__name__)

# TODO: Stream audio
# TODO: fp8 quantization for inference

class TTS:
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        self.lm_engine = VLLMEngine(model_path, device, gpu_memory_utilization=0.4)
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
            temperature=0.5,
            top_k=15,
            stop_token_ids=self.text_tokenizer.stop_token,
            max_tokens=1024,
            logits_processors=logits_processors
        )

    async def generate_async(self,
        text: str,
        speaker: str,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if speaker is None:
            raise ValueError('Speaker is required')

        start_time = time.time()
        batch_text = sanitize_text(text)
        input_tokens = [self.text_tokenizer.prepare_tts_tokens(text, speaker) for text in batch_text]

        logger.info(f'Texts after preprocessing: {batch_text}, {speaker}', extra={'request_id': request_id})
        logger.info(f'Input tokens shape: {len(input_tokens)} and batch size: {len(batch_text)}', extra={'request_id': request_id})

        prompt = TokensPrompt(prompt_token_ids=input_tokens[0])
        
        results_generator = self.lm_engine.engine.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            request_id=request_id
        )

        preds: List[RequestOutput] = []

        async for request_output in results_generator:
            if request_output.finished:
                preds.append(request_output)

        mimi_tokens = []

        metrics = {
            'time_to_first_token': [],
            'time_to_last_token': [],
            'input_tokens': [],
            'decoding_tokens': []
        }

        for idx, request_output in enumerate(preds):
            o = np.array(request_output.outputs[0].token_ids)
            end = np.where(o == self.text_tokenizer.stop_token[0])[0]

            if len(end) > 0:
                end = end[0]
            else:
                end = len(o)

            o = o[:end]
            o = o - cfg.OFFSET[MIMI]
            o = deserialize_tokens(o, cfg.n_codebooks)
            assert np.all(o >= 0), f'Negative token index generated for batch {idx}'

            metrics['time_to_first_token'].append(
                request_output.metrics.first_token_time - request_output.metrics.first_scheduled_time
            )
            metrics['time_to_last_token'].append(
                request_output.metrics.finished_time - request_output.metrics.first_scheduled_time
            )
            metrics['input_tokens'].append(len(request_output.prompt_token_ids))
            metrics['decoding_tokens'].append(len(request_output.outputs[0].token_ids))

            mimi_tokens.append(o)

        mimi_tokens = np.concatenate(mimi_tokens, axis=1)
        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        audio, decode_time = self.audio_tokenizer.decode(mimi_tokens)
        audio = pyln.normalize.peak(audio, -1.0)

        metrics = TTSMetrics(
            time_to_first_token=metrics['time_to_first_token'],
            time_to_last_token=metrics['time_to_last_token'],
            time_to_decode_audio=decode_time,
            input_tokens=metrics['input_tokens'],
            decoding_tokens=metrics['decoding_tokens'],
            generate_end_to_end_time=time.time()-start_time
        )

        return {"audio": audio, "metrics": metrics}

async def main():
    import uuid
    import torchaudio

    model = TTS('cmeraki/mimi_tts_hf_stage', 'cuda:0')
    result = await model.generate_async(
        'मेरे प्यारे देशवासियों, आज हम एक नए भारत की ओर कदम बढ़ा रहे हैं।',
        speaker='[spkr_youtube_webds_hi_pmmodi]',
        request_id=str(uuid.uuid4())
    )

    torchaudio.save('output.wav', torch.from_numpy(result['audio']), sample_rate=24000, format='mp3')

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
