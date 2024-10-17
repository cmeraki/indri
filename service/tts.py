import sys
sys.path.append('omni/')

import torch
import numpy as np
from transformers import MimiModel
from vllm import LLM, SamplingParams
from typing import List
from functools import partial

from commons import TEXT, MIMI, CONVERT
from commons import Config as cfg
from omni.train_with_mimi import get_text_tokenizer

import service.utils as utils
from .logger import get_logger

logger = get_logger(__name__)

# DONE: Add logit processor for vLLM
# TODO: Try out Async engine for streaming audio
# TODO: fp8 quantization for inference
# TODO: vLLM server for best performance
# TOOD: Expose error messages - Decoding errors or model errors

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
            dtype='float16'
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

        self.sampling_params = SamplingParams(
            temperature=0.4,
            top_k=100,
            stop_token_ids=self.stop_token,
            max_tokens=1024,
            logits_processors=[partial(utils.alternative_logits_processor, **logits_processor_kwargs)]
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
        speaker='[spkr_hifi_tts_9017]'
    ):
        batch_text = utils.sanitize_text(text)

        logger.debug(f'Texts after preprocessing: {batch_text}')
        input_tokens = [self.prepare_tokens(text, speaker) for text in batch_text]
        logger.info(f'Input tokens shape: {sum([len(t) for t in input_tokens])}')

        out = self.model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=self.sampling_params
        )

        mimi_tokens = []
        for o in out:
            o = o.outputs[0].token_ids
            o = np.array(o)
            end = np.where(o == self.stop_token[0])[0]
            if len(end) > 0:
                end = end[0]
            else:
                end = len(o)
            o = o[:end]

            o = o - cfg.OFFSET[MIMI]
            mimi_tokens.extend(o)

        mimi_tokens = utils.deserialize_tokens(np.array(mimi_tokens))
        mimi_tokens = torch.tensor(np.expand_dims(mimi_tokens, axis=0), device=self.device)

        logger.info(f'Mimi tokens shape: {mimi_tokens.shape}')

        with torch.no_grad():
            audio = self.audio_tokenizer.decode(mimi_tokens).audio_values
            audio = audio.detach().cpu().numpy()

        logger.info(f'Audio shape: {audio.shape}')

        return audio

if __name__ == '__main__':
    tts = TTS('cmeraki/mimi_tts_hf', 'cuda:0')
    audio = tts.generate('Hello, how are you?')
