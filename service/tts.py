import numpy as np
from transformers import MimiModel
from vllm import LLM, SamplingParams

from omni.config import cfg
from omni.logger import get_logger
from omni.train import get_text_tokenizer

logger = get_logger(__name__)

class TTS:
    def __init__(self, model_path):
        self.audio_tokenizer = MimiModel.from_pretrained("kyutai/mimi")

        self.model = LLM(
            model=model_path,
            skip_tokenizer_init=True,
            gpu_memory_utilization=0.6,
            dtype='float32'
        )

        self.text_tokenizer = get_text_tokenizer()

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_k=100,
            stop_token_ids=self.stop_token
        )

    def prepare_tokens(self, incoming_text, speaker) -> List[int]:
        incoming_tokens = text_tokenizer.encode(incoming_text)

        input_tokens = np.hstack([
            dl.text_modality_token,
            incoming_tokens,
            dl.convert_token,
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker)
        ])

        return input_tokens.tolist()

    def generate_batch(self, batch_text: List[str], speaker='[spkr_hifi_tts_9017]'):
        logger.info(f'Texts after preprocessing: {batch_text}')
        input_tokens = [prepare_tokens(text, speaker) for text in batch_text]
        logger.info(f'Batch after preprocessing: {batch_text}')

        outputs = self.model.generate(
            input_tokens,
            self.sampling_params
        )

        mimi_tokens = []

        for o in outputs:
            o = o.outputs[0].token_ids
            logger.info(f'Mimi tokens: {self.text_tokenizer.decode(o)}')

            o = o - cfg.OFFSET[MIMI]
            mimi_tokens.extend(o)

        mimi_tokens = deserialize_tokens(mimi_tokens)

        return mimi_tokens