import sys
sys.path.append('omni/')

import time
import torch
import numpy as np
from transformers import MimiModel, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from functools import partial
import pyloudnorm as pyln

from vllm import SamplingParams, AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.inputs import TokensPrompt

from commons import TEXT, MIMI, CONVERT
from commons import Config as cfg

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
    def __init__(self, model_path, device):
        self.device = device
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device=self.device)
        self.audio_tokenizer.eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.engine_args = AsyncEngineArgs(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.8,
            dtype='bfloat16'
        )
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])

        logits_processor_kwargs = {
            'num_codebooks': cfg.n_codebooks,
            'codebook_size': cfg.per_codebook_size,
            'offset': cfg.OFFSET[MIMI],
            'stop_token': self.stop_token
        }
        logits_processors = [
            partial(alternative_logits_processor, **logits_processor_kwargs)
        ]

        self.sampling_params = SamplingParams(
            temperature=0.5,
            top_k=15,
            stop_token_ids=self.stop_token,
            max_tokens=1024,
            logits_processors=logits_processors
        )

    def prepare_tokens(self, incoming_text, speaker) -> List[int]:
        incoming_tokens = self.text_tokenizer.encode(incoming_text)

        input_tokens = np.hstack([
            self.text_modality_token,
            incoming_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self.text_tokenizer.encode(speaker)
        ])

        return input_tokens.tolist()

    async def generate_async(self,
        text: str,
        speaker: str,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if speaker is None:
            raise ValueError('Speaker is required')

        start_time = time.time()
        batch_text = sanitize_text(text)
        input_tokens = [self.prepare_tokens(text, speaker) for text in batch_text]

        logger.info(f'Texts after preprocessing: {batch_text}, {speaker}', extra={'request_id': request_id})
        logger.info(f'Input tokens shape: {len(input_tokens)} and batch size: {len(batch_text)}', extra={'request_id': request_id})

        prompt = TokensPrompt(prompt_token_ids=input_tokens[0])

        results_generator = self.engine.generate(
            prompt=prompt,
            sampling_params=self.sampling_params,
            request_id=request_id
        )

        preds = []

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
            end = np.where(o == self.stop_token[0])[0]
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

            mimi_tokens.append(o)

        mimi_tokens = np.concatenate(mimi_tokens, axis=1)
        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        audio, decode_time = self.decode_audio(mimi_tokens)
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

    def decode_audio(self, audio_tokens) -> Tuple[np.ndarray, float]:
        start_time = time.time()

        with torch.no_grad():
            audio_tokens = torch.tensor(np.expand_dims(audio_tokens, axis=0), device=self.device)
            audio = self.audio_tokenizer.decode(audio_tokens).audio_values
            audio = audio.detach().cpu().numpy()[0]

        end_time = time.time()
        logger.info(f'Time taken to decode audio: {end_time - start_time} seconds')

        return audio, end_time - start_time

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
