import numpy as np
import encodec
import torch
from encodec.utils import convert_audio, save_audio

from tokenize_audio import get_model
model = get_model(bandwidth=6)
model.segment = 1

# tokenize audio
# 1. Add start end tokens
# 2. Create large files with one dataset each
# 3. Don't use all of tokens for training the llm. save llm for first 2 encodec layers and use another network for the rest.
#    recreate audio using suno's last model
# 4. check recreation of suno from top 2 codebooks
# 5.

# check recreation from top 2 codebooks - works

import bark

def load(file_paths):
    for i, f in enumerate(file_paths):
        x = np.load(f)
        good_audio = bark.api.generate_fine(x_coarse_gen=x[0, 0:2, :], silent=False)
        print(good_audio.shape)
        good_audio = np.expand_dims(good_audio, axis=0)
        good_audio = torch.from_numpy(good_audio)
        wav = model.decode([(good_audio, None)])
        save_audio(wav[0], 'test.wav', sample_rate=24000)
        break

import glob
load(glob.glob("data/audio_tokens/*.npy"))