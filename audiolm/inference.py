from tokenize_audio import get_model, START_TOKEN
import numpy as np
def deserialize_tokens(tokens: np.ndarray):
    # serial token shape = n,1
    # deserialize to (codebook, tokens)
    # remove start_token
    start_indices = tokens == START_TOKEN
    splits = np.split(tokens, indices_or_sections=start_indices)
    splits = [split.reshape((tokens.shape[0] // 2, 2), order='F') for split in splits]
    return splits


def test_generation(tokens):
    model = get_model(bandwidth=3)
    deserialize_tokens(tokens)
    model.decode(encoded_frames=tokens)


if __name__ == "__main__":
    tokens = np.load('data/audio_tokens/8424804c-679d-4cc0-a265-5f4e50ac3f87.npy')
    test_generation(tokens)