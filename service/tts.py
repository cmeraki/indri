import os
import torch
import numpy as np
from transformers import MimiModel
from vllm import LLM, SamplingParams
from typing import List

from commons import CTX, TEXT, MIMI, CONVERT
from commons import Config as cfg
from omni.logger import get_logger
from omni.train_with_mimi import get_text_tokenizer
import service.utils as utils

logger = get_logger(__name__)

class TTS:
    def __init__(self, model_path, device):
        self.device = device
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device=self.device)
        self.audio_tokenizer.eval()
        self.text_tokenizer = get_text_tokenizer()

        self.model = LLM(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.9,
            dtype='float16'
        )

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])

        self.sampling_params = SamplingParams(
            temperature=0.4,
            top_k=100,
            stop_token_ids=self.stop_token,
            max_tokens=1024
        )

    def preprocess_text(self, text: str) -> List[str]:
        text = utils.normalize_text(text)
        text = utils.split_and_join_sentences(text)

        return text

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
        speaker='[spkr_hifi_tts_9017]',
        batch_size=32
    ):
        batch_text = self.preprocess_text(text)

        logger.debug(f'Texts after preprocessing: {batch_text}')
        input_tokens = [self.prepare_tokens(text, speaker) for text in batch_text]
        logger.debug(f'Input tokens: {input_tokens} and batch size: {batch_size}')

        out = self.model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=self.sampling_params
        )

        mimi_tokens = []
        for o in out:
            o = o.outputs[0].token_ids
            o = np.array(o)
            end = np.where(o == self.stop_token[0])[0][0]
            o = o[:end]

            o = o - cfg.OFFSET[MIMI]
            mimi_tokens.extend(o)

        mimi_tokens = utils.deserialize_tokens(np.array(mimi_tokens))
        mimi_tokens = torch.tensor(np.expand_dims(mimi_tokens, axis=0), device=self.device)

        with torch.no_grad():
            audio = self.audio_tokenizer.decode(mimi_tokens).audio_values
            audio = audio.detach().cpu().numpy()

        return audio

if __name__ == '__main__':
    tts = TTS('cmeraki/mimi_tts_hf', 'cuda:0')
    audio = tts.generate('Hello, how are you?')
