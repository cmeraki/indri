import sys
sys.path.append('omni/')

import time
import torch
import numpy as np
from transformers import MimiModel
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple, Optional
from functools import partial

from commons import TEXT, MIMI, CONVERT
from commons import Config as cfg
from omni.train_with_mimi import get_text_tokenizer

import service.utils as utils
from .logger import get_logger
from .models import TTSMetrics

print(time.time())
logger = get_logger(__name__)

# DONE: Add logit processor for vLLM
# TODO: Try out Async engine for streaming audio
# TODO: fp8 quantization for inference
# TODO: vLLM server for best performance
# DONE: Expose error messages - Decoding errors or model errors

class TTS:
    def __init__(self, model_path, device):
        self.device = device
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device=self.device)
        self.audio_tokenizer.eval()
        self.text_tokenizer = get_text_tokenizer()

        self.model = LLM(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.8,
            dtype='bfloat16'
        )

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])

        logits_processor_kwargs = {
            'n_codebooks': cfg.n_codebooks,
            'per_codebook_size': cfg.per_codebook_size,
            'offset': cfg.OFFSET[MIMI]
        }
        logits_processors = [
            partial(utils.alternative_logits_processor, **logits_processor_kwargs)
        ]

        self.sampling_params = SamplingParams(
            temperature=0.4,
            top_k=100,
            stop_token_ids=self.stop_token,
            max_tokens=1024,
            # logits_processors=logits_processors
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

    def generate(
        self,
        text: str,
        speaker: Optional[str] = '[spkr_hifi_tts_9017]'
    ) -> Dict[str, Any]:
        start_time = time.time()
        batch_text = utils.sanitize_text(text)

        logger.info(f'Texts after preprocessing: {batch_text}, {speaker}')
        input_tokens = [self.prepare_tokens(text, speaker) for text in batch_text]
        logger.info(f'Input tokens shape: {sum([len(t) for t in input_tokens])} and batch size: {len(batch_text)}')

        try:
            preds = self.model.generate(
                prompt_token_ids=input_tokens,
                sampling_params=self.sampling_params
            )
        except Exception as e:
            logger.error(f'Error in generating tokens: {e}')
            raise RuntimeError(f'Error in generating tokens')

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
            o = utils.deserialize_tokens(o)
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
            audio = audio.detach().cpu().numpy()

        end_time = time.time()
        logger.info(f'Time taken to decode audio: {end_time - start_time} seconds')

        return audio, end_time - start_time

if __name__ == '__main__':
    tts = TTS('cmeraki/mimi_tts_hf', 'cuda:0')
    result = tts.generate('Hello, how are you?')

    print(result['metrics'])
