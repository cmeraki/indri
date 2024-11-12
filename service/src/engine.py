import time
import torch
import numpy as np
from typing import Tuple, List

from vllm import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from transformers import MimiModel, AutoTokenizer

from ..logger import get_logger
from ..utils import codebook_encoding

import sys
sys.path.append('omni/')
from commons import TEXT, MIMI, CONVERT
from commons import Config as cfg

logger = get_logger(__name__)


# TODO: Engine will be deprecated soon.
# We will use the VLLMEngine class for all tasks since a single engine can handle all tasks.
class Engine:
    """Wrapper class for the VLLM engine"""
    def __init__(self, model_path: str, device: str = 'cuda:0', gpu_memory_utilization: float = 0.8):
        engine_args = AsyncEngineArgs(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            dtype='bfloat16',
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    @property
    def engine(self) -> AsyncLLMEngine:
        return self._engine


class VLLMEngine:
    """Singleton class for the VLLM engine"""
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine(*args, **kwargs)
        return cls._instance

    def _init_engine(cls, model_path: str, device: str = 'cuda:0', gpu_memory_utilization: float = 0.8):
        engine_args = AsyncEngineArgs(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            dtype='bfloat16',
        )
        cls._engine = AsyncLLMEngine.from_engine_args(engine_args)

    @property
    def engine(cls) -> AsyncLLMEngine:
        return cls._engine

class AudioTokenizer:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = 'cuda:0'):
        self.device = device

        self._tokenizer = MimiModel.from_pretrained("kyutai/mimi").to(device=self.device)
        self._tokenizer.eval()

    def decode(self, audio_tokens: np.ndarray) -> Tuple[np.ndarray, float]:
        start_time = time.time()

        with torch.no_grad():
            audio_tokens = torch.tensor(np.expand_dims(audio_tokens, axis=0), device=self.device)
            audio = self._tokenizer.decode(audio_tokens).audio_values

        audio = audio.detach().cpu().numpy()[0]
        end_time = time.time()

        return audio, end_time - start_time

    def encode(self, audio: torch.Tensor) -> Tuple[np.ndarray, float]:
        start_time = time.time()

        with torch.no_grad():
            audio = audio.to(device=self.device)
            tokens = self._tokenizer.encode(audio, num_quantizers=cfg.n_codebooks).audio_codes

        tokens = codebook_encoding(
            tokens.detach().cpu().numpy()[0],
            cfg.per_codebook_size,
            cfg.OFFSET[MIMI]
        )
        end_time = time.time()
        return tokens, end_time - start_time

class TextTokenizer:
    def __init__(self, model_path: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.convert_token = self._tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self._tokenizer.encode(cfg.STOP_TOKEN)
        self.text_modality_token = self._tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self._tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])

    def decode(self, tokens: np.ndarray) -> str:
        return self._tokenizer.decode(tokens)

    def prepare_tts_tokens(self, text: str, speaker: str) -> List[int]:
        incoming_tokens = self._tokenizer.encode(text)

        input_tokens = np.hstack([
            self.text_modality_token,
            incoming_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self._tokenizer.encode(speaker)
        ])

        return input_tokens.tolist()

    def prepare_asr_tokens(self, audio_tokens: np.ndarray) -> List[int]:
        input_tokens = np.hstack([
            self.acoustic_modality_token,
            audio_tokens,
            self.convert_token,
            self.text_modality_token
        ])

        return input_tokens.tolist()

    def prepare_audio_continuation_tokens(self, audio_tokens: np.ndarray) -> List[int]:
        input_tokens = np.hstack([
            self.acoustic_modality_token,
            audio_tokens
        ])

        return input_tokens.tolist()
