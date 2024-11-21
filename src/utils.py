import re
import torch
import numpy as np
from typing import Tuple

from .logger import get_logger
logger = get_logger(__name__)

# TODO: Rewrite in pytorch
def deserialize_tokens(tokens, num_codebooks):
    cb = [tokens[i::num_codebooks] for i in range(num_codebooks)]
    min_shape = min([c.shape for c in cb])[0]
    acoustic_tokens = np.stack([c[:min_shape] - 2048 * i for i, c in enumerate(cb)])
    return acoustic_tokens

# TODO: Rewrite in pytorch
def codebook_encoding(tokens: torch.tensor, per_codebook_size: int, offset: int):
    """Receive n/4 x 4, flatten, add offset"""

    c, n = tokens.shape

    # Adding codebook offset
    for i in range(c):
        tokens[i, :] += i * per_codebook_size

    flat_arr = tokens.reshape(c * n, order='F')
    flat_arr += offset

    return flat_arr

def sanitize_text(text: str) -> list[str]:
    """
    Sanitize text to be used for TTS

    Args:
        text (str): Text to sanitize

    Returns:
        list[str]: List of sentences, split by punctuation (., !, ?)
    """
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # allowed_pattern = r'[^a-z0-9\s,\.?\n\!]'
    # text = re.sub(allowed_pattern, '', text)
    text = re.sub(r'([,\.?])+', r'\1', text)

    # pattern = r'([.!?])'
    # segments = re.split(pattern, text)

    sentences = [text.strip()]
    return sentences

    # current_sentence = ''

    for segment in segments:
        current_sentence += segment
        if segment in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ''

    if current_sentence:
        sentences.append(current_sentence.strip())

    return [s for s in sentences if s]


def alternative_logits_processor(
        past_token_ids: Tuple,
        logits: torch.Tensor,
        num_codebooks: int,
        codebook_size: int,
        offset: int,
        stop_token: int
    ) -> torch.Tensor:
    """
    Logits processor for alternating codebooks
    Given a sequence of logits, we want to make sure that the alternating tokens
    are chosen from different codebooks.

    Args:
        past_token_ids: Tuple[int] - Tuple of past token ids
        logits: torch.Tensor - Logits to process. Shape (vocab_size)

    Returns:
        torch.Tensor - Processed logits. Shape (vocab_size)
    """
    logger.debug(f'Stop token: {stop_token}')

    codebook_indices = len(past_token_ids) % num_codebooks

    start_idx = offset + codebook_indices * codebook_size
    end_idx = offset + (codebook_indices + 1) * codebook_size

    logger.debug(f'Past_token_ids: {len(past_token_ids)}, codebook indices: {codebook_indices}, start idx: {start_idx}, end idx: {end_idx}')

    mask = torch.zeros_like(logits)
    mask[start_idx:end_idx] = 1
    mask[stop_token] = 1

    new_logits = logits.clone()
    new_logits = new_logits * mask
    new_logits[mask == 0] = -torch.inf

    return new_logits

if __name__ == '__main__':
    print(
        alternative_logits_processor(
            (1, 2, 3, 4),
            torch.rand(100),
            num_codebooks=4,
            codebook_size=20,
            offset=10,
            stop_token=-1
        )
    )

    print(
        alternative_logits_processor(
            (),
            torch.rand(100),
            num_codebooks=4,
            codebook_size=20,
            offset=10,
            stop_token=-1
        )
    )

    sm = torch.nn.Softmax(dim=-1)
    out = alternative_logits_processor(
        (1, 2, 3, 4),
        torch.rand(100),
        num_codebooks=4,
        codebook_size=20,
        offset=10,
        stop_token=-1
    )
    out = sm(out)

    print(out)

