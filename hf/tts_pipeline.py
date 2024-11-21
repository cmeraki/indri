import re
import torch
import numpy as np
from transformers import MimiModel, GenerationConfig
from transformers import Pipeline

class IndriTTSPipeline(Pipeline):
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
        speaker = kwargs.get('speaker', '[spkr_unk]')

        preprocess_kwargs = {
            'speaker': speaker
        }

        return preprocess_kwargs, {}, {}

    def _prepare_tts_tokens(self, text_tokens, speaker):
        input_tokens = np.hstack([
            self.text_modality_token,
            text_tokens,
            self.convert_token,
            self.acoustic_modality_token,
            self.tokenizer.encode(speaker)
        ])

        return input_tokens.tolist()

    def _sanitize_text(self, text):
        text = text.lower()
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)

        text = re.sub(r'([,\.?])+', r'\1', text)

        return text.strip()

    def _deserialize_tokens(self, tokens, num_codebooks):
        cb = [tokens[i::num_codebooks] for i in range(num_codebooks)]
        min_shape = min([c.shape for c in cb])[0]
        acoustic_tokens = torch.vstack([c[:min_shape] - 2048*i for i, c in enumerate(cb)])

        return acoustic_tokens

    def preprocess(self, inputs, speaker):
        # TODO: Check for batching
        input_text = self._sanitize_text(inputs)
        input_tokens = self.tokenizer.encode(input_text)
        task_tokens = self._prepare_tts_tokens(input_tokens, speaker)
        task_tokens = torch.tensor(task_tokens).unsqueeze(0)

        return {'task_tokens': task_tokens}

    def _forward(self, model_inputs, **forward_args):

        outputs = self.model.generate(model_inputs['task_tokens'])
        audio_tokens = []

        for idx, inputs in enumerate(model_inputs['task_tokens']):
            truncated = outputs[idx, inputs.shape[-1]:]
            end = torch.where(truncated == self.stop_token[0])[-1]
    
            if end.shape[-1] > 0:
                end = end[0]
            else:
                end = truncated.shape[-1]
    
            truncated = truncated[:end]
            truncated -= self.audio_offset
            truncated = self._deserialize_tokens(torch.tensor(truncated), self.num_codebooks)
            audio_tokens.append(truncated)

        audio_tokens = torch.vstack(audio_tokens).unsqueeze(0)
        audio = self.audio_tokenizer.decode(audio_tokens).audio_values

        return {
            'audio_tokens': audio_tokens, # (B, num_codebooks, num_samples)
            'audio': audio # (B, 1, num_audio_samples)
        }

    def postprocess(self, model_outputs):
        return model_outputs