import re
import torch
import torchaudio
import numpy as np
from transformers import MimiModel, GenerationConfig
from transformers import Pipeline

class IndriASRPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.audio_tokenizer = MimiModel.from_pretrained('kyutai/mimi').to(device=self.device)

        # TODO: Ideally all of this should come from model config
        self.convert_token = self.tokenizer.encode('[convert]')
        self.stop_token = self.tokenizer.encode('[stop]')
        self.text_modality_token = self.tokenizer.encode('[text]')
        self.acoustic_modality_token = self.tokenizer.encode('[mimi]')
        self.num_codebooks = 8
        self.audio_offset = 50257

        self.model.generation_config = GenerationConfig(
            eos_token_id=self.stop_token,
            max_length=kwargs.get('max_length', 1024),
            temperature=kwargs.get('temperature', 0.5),
            top_k=kwargs.get('top_k', 15),
            do_sample=kwargs.get('do_sample', True)
        )

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def _prepare_asr_tokens(self, audio_tokens):
        input_tokens = np.hstack([
            self.acoustic_modality_token,
            audio_tokens,
            self.convert_token,
            self.text_modality_token
        ])

        return input_tokens.tolist()

    def _sanitize_text(self, text):
        text = text.lower()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)

        text = re.sub(r'([,\.?])+', r'\1', text)

        return text.strip()

    def codebook_encoding(tokens: torch.tensor, per_codebook_size: int, offset: int):
        c, n = tokens.shape

        for i in range(c):
            tokens[i, :] += i * per_codebook_size

        flat_arr = tokens.reshape(c * n, order='F')
        flat_arr += offset

        return flat_arr

    def preprocess(self, inputs):
        # TODO: Check for batching
        audio, sample_rate = inputs['audio'], inputs['sample_rate']
        audio = torchaudio.transforms.Resample(sample_rate, 24000)(audio)

        return {'audio': audio}

    def _forward(self, model_inputs, **forward_args):

        audio_tokens = self.audio_tokenizer.encode(
            model_inputs['audio'],
            num_quantizers=self.num_codebooks
        ).audio_codes

        audio_tokens = self._codebook_encoding(
            audio_tokens,
            self.num_codebooks,
            self.audio_offset
        )

        input_tokens = self._prepare_asr_tokens(audio_tokens)
        task_tokens = torch.tensor(input_tokens).unsqueeze(0)

        outputs = self.model.generate(task_tokens)
        text_tokens = []

        for idx, inputs in enumerate(model_inputs['task_tokens']):
            truncated = outputs[idx, inputs.shape[-1]:]
            end = torch.where(truncated == self.stop_token[0])[-1]
    
            if end.shape[-1] > 0:
                end = end[0]
            else:
                end = truncated.shape[-1]
    
            truncated = truncated[:end]
            text_tokens.append(truncated)

        text_tokens = torch.vstack(text_tokens).unsqueeze(0)
        text = self.tokenizer.decode(text_tokens)

        return {
            'text_tokens': text_tokens,
            'text': text
        }

    def postprocess(self, model_outputs):
        return model_outputs
