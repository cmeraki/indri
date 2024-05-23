import numpy as np
import encodec
import torch
from encodec.utils import convert_audio, save_audio

from encodec_try import get_model
model = get_model(bandwidth=6)
model.segment = 1

def load(file_paths):
    for i, f in enumerate(file_paths):
        x = np.load(f)
        x = torch.from_numpy(x)
        x = x.reshape(shape=(x.shape[0], -1))
        print(x.shape)

import glob
load(glob.glob("data/audio_tokens/*.npy"))