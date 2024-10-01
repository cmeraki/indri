import torch
import os

import numpy as np
from transformers import MimiModel, AutoFeatureExtractor, AutoTokenizer

from configs.constants import *

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

class TextTokenizer:
    def __init__(self, name='cmeraki/gpt2-124M-400B'):
        self.type = TEXT
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        print("text vocab size", self.tokenizer.vocab_size)

    def encode(self, text: str):
        tokens = self.tokenizer.encode(text)
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class MimiTokenizer:
    def __init__(self, device):    
        self.device = device
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        # self.model = torch.compile(self.model)
        self.model.to(device)
        self.model.eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi", device=device)
        self.sampling_rate = self.feature_extractor.sampling_rate
        self.n_codebooks = 8
        self.vocab_size = 2048

    @torch.inference_mode()
    def encode(self, waveform):
        inputs = self.feature_extractor(raw_audio=waveform, 
                                        sampling_rate=self.sampling_rate, 
                                        return_tensors="pt").to(self.device)
            
        output = self.model.encode(inputs["input_values"], inputs["padding_mask"], num_quantizers=self.n_codebooks)
        tokens = output.audio_codes[0].cpu().numpy()
        return tokens

    def decode(self, tokens):
        assert len(tokens.shape) == 2
        tokens = torch.tensor(np.expand_dims(tokens, axis=0)).to(self.device)
        output = self.model.decode(tokens)
        waveform = output.audio_values.cpu()
        return waveform


def flatten_tokens(arr: torch.tensor,
                   per_codebook_size: int):
    
    c, n = arr.shape
    i_values = np.arange(c) * per_codebook_size
    arr += i_values.reshape(c, 1)
    flat_arr = arr.reshape(c * n, order='F')
    return flat_arr


def deflatten_tokens(tokens, n_codebooks, per_codebook_size):
    arr = []
    for i in range(n_codebooks):
        arr[i] = tokens[i::n_codebooks] - per_codebook_size * i
    
    # min_shape = min(cb1.shape, cb2.shape, cb3.shape, cb4.shape)[0]
    acoustic_tokens = np.stack(arr)
    return acoustic_tokens

def get_tokenizer(type, device):
    tokenizer = None
    if type == MIMI:
        tokenizer = MimiTokenizer(device=device)

    if type == TEXT:
        tokenizer = TextTokenizer()

    return tokenizer



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    
    args = parser.parse_args()
    tokenizer = MimiTokenizer(device='cuda:0')
    
    from datasets import load_dataset, Audio
    librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=tokenizer.sampling_rate))
    audio_sample = librispeech_dummy[-1]["audio"]["array"]
    print(audio_sample.shape)
    from tqdm import tqdm
    for i in tqdm(range(10000)):
        tokens = tokenizer.encode(audio_sample)
    print(tokens.shape)

    audio = tokenizer.decode(tokens)
    print(audio)

    import torchaudio
    torchaudio.save('test.wav', audio.reshape(1, -1), sample_rate=tokenizer.sampling_rate, compression=torchaudio.io.CodecConfig(bit_rate=128000))

