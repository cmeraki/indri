from tokenize_audio import get_model, START_TOKEN
import numpy as np
import bark
import torch
from encodec.utils import save_audio

def deserialize_tokens(tokens: np.ndarray, n_channels=2):
    # serial token shape = n,1
    # deserialize to (codebook, tokens)
    # remove start_token
    start_indices = tokens == START_TOKEN
    start_indices = np.argwhere(start_indices).reshape(-1)
    start_indices = start_indices[1:]
    splits = np.split(tokens, indices_or_sections=start_indices)
    codebook_deindex = np.arange(n_channels) * 1024
    codebook_deindex = np.expand_dims(codebook_deindex, axis=-1)
    splits = [split[1:].reshape((2, split[1:].shape[0] // 2), order='F') - codebook_deindex for split in splits]
    return splits


def test_generation(tokens):
    model = get_model(bandwidth=3)
    tokens = deserialize_tokens(tokens)
    token_single = np.expand_dims(tokens[0], axis=0)
    good_audio = bark.api.generate_fine(x_coarse_gen=token_single[0, 0:2, :], silent=False)
    good_audio = np.expand_dims(good_audio, axis=0)
    good_audio = torch.from_numpy(good_audio)
    wav = model.decode([(good_audio, None)])
    save_audio(wav[0], 'test.wav', sample_rate=24000)

if __name__ == "__main__":
    tokens = np.load('data/audio_tokens/cbbc0606-7206-4f00-b73e-e9692bf70be0.npy')
    test_generation(tokens)