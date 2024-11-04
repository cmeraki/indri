import sys
sys.path.append('omni/')

import time
import torch
import numpy as np
from transformers import MimiModel, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
from functools import partial

from vllm import SamplingParams, AsyncEngineArgs, RequestOutput
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

    def prepare_tokens(self, incoming_text, speaker, prompt_tokens: dict = None) -> List[int]:
        incoming_tokens = self.text_tokenizer.encode(incoming_text)

        if prompt_tokens:
            input_tokens = np.hstack([
                self.text_modality_token,
                prompt_tokens[TEXT],
                incoming_tokens,
                self.convert_token,
                self.acoustic_modality_token,
                self.text_tokenizer.encode(speaker),
                prompt_tokens[MIMI],
            ])
            return input_tokens.tolist()

        input_tokens = np.hstack([
            self.text_modality_token,
            incoming_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self.text_tokenizer.encode(speaker)
        ])

        return input_tokens.tolist()

    def get_generation_output(self, output: List[RequestOutput]) -> Tuple[np.ndarray, Dict[str, Any]]:
        output_tokens = []

        metrics = {
            'time_to_first_token': [],
            'time_to_last_token': [],
            'input_tokens': [],
            'decoding_tokens': []
        }

        for idx, request_output in enumerate(output):
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

            output_tokens.append(o)

        output_tokens = np.concatenate(output_tokens, axis=1)
        return output_tokens, metrics

    async def generate_async(self,
        text: str,
        speaker: Optional[str] = '[spkr_hifi_tts_9017]',
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:

        start_time = time.time()
        batch_text = sanitize_text(text)
        prompt_tokens = {}
        overall_metrics = []
        mimi_tokens = []

        logger.info(f'Texts after preprocessing: {batch_text}, {speaker}', extra={'request_id': request_id})

        for text in batch_text:
            input_tokens = self.prepare_tokens(text, speaker, prompt_tokens)
            logger.info(f'Input tokens shape: {len(input_tokens)}', extra={'request_id': request_id})
            prompt = TokensPrompt(prompt_token_ids=input_tokens)

            results_generator = self.engine.generate(
                prompt=prompt,
                sampling_params=self.sampling_params,
                request_id=request_id
            )

            async for request_output in results_generator:
                if request_output.finished:
                    output = request_output

            output_tokens, generation_metrics = self.get_generation_output(output=output)
            logger.info(f'Output tokens shape: {output_tokens.shape}', extra={'request_id': request_id})

            overall_metrics.append(generation_metrics)
            mimi_tokens.append(output_tokens)

            prompt_tokens = {
                TEXT: input_tokens,
                MIMI: output_tokens
            }

        audio, decode_time = self.decode_audio(mimi_tokens)

        metrics = TTSMetrics(
            time_to_first_token=overall_metrics[0]['time_to_first_token'],
            time_to_last_token=sum([x['time_to_last_token'] for x in overall_metrics]),
            time_to_decode_audio=decode_time,
            input_tokens=[x['input_tokens'] for x in overall_metrics],
            decoding_tokens=[x['decoding_tokens'] for x in overall_metrics],
            generate_end_to_end_time=time.time()-start_time
        )

        return {"audio": audio, "metrics": metrics}

    def decode_audio(self, audio_tokens) -> Tuple[np.ndarray, float]:
        start_time = time.time()

        with torch.no_grad():
            audio_tokens = torch.tensor(np.expand_dims(audio_tokens, axis=0), device=self.device)
            audio = self.audio_tokenizer.decode(audio_tokens).audio_values
            audio = audio.detach().cpu().numpy()

        end_time = time.time()
        logger.info(f'Time taken to decode audio: {end_time - start_time} seconds')

        return audio, end_time - start_time

if __name__ == '__main__':
    tts = TTS('cmeraki/mimi_tts_hf', 'cuda:0')
    result = tts.generate('Long ago, in a distant kingdom between emerald hills and sapphire lakes, magic flowed freely. A wise king ruled, ensuring peace and prosperity.')

    print(result['metrics'])
