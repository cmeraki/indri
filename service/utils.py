import re
import torch
import numpy as np
from typing import Tuple

from .logger import get_logger
logger = get_logger(__name__)

def normalize_text(text):
    text = text.lower()
    text = text.replace("<comma>", ',')
    text = text.replace("<period>", '.')
    text = text.replace('<questionmark>', '?')
    text = text.replace('<exclamationpoint>', '!')
    text = text.replace("\n", " ")
    return text


def deserialize_tokens(tokens):
    cb1 = tokens[0::4]
    cb2 = tokens[1::4]
    cb3 = tokens[2::4]
    cb4 = tokens[3::4]

    min_shape = min(cb1.shape, cb2.shape, cb3.shape, cb4.shape)[0]
    acoustic_tokens = np.stack([cb1[:min_shape], cb2[:min_shape] - 2048, cb3[:min_shape] - 4096, cb4[:min_shape] - 6144])

    return acoustic_tokens


def split_and_join_sentences(text):
    pattern = r'([.!?])'
    segments = re.split(pattern, text)

    sentences = []
    current_sentence = ''

    for segment in segments:
        current_sentence += segment
        if segment in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ''

    if current_sentence:
        sentences.append(current_sentence.strip())

    return [s for s in sentences if s]


def alternative_logits_processor(past_token_ids: Tuple, logits: torch.Tensor, **kwargs) -> torch.Tensor:
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
    num_codebooks = kwargs.get('n_codebooks', 4)
    codebook_size = kwargs.get('per_codebook_size', 2048)
    offset = kwargs.get('offset', 50257)

    new_logits = logits.clone()
    codebook_indices = len(past_token_ids) % num_codebooks

    logger.info(f'Logits shape: {logits.shape}, past_token_ids: {len(past_token_ids)}, codebook indices: {codebook_indices}')

    mask = torch.zeros_like(new_logits)
    start_idx = offset + codebook_indices * codebook_size
    end_idx = offset + (codebook_indices + 1) * codebook_size

    logger.info(f'Start idx: {start_idx}, end idx: {end_idx}')

    mask[start_idx:end_idx] = 1
    new_logits = new_logits * mask

    return new_logits

if __name__ == '__main__':
    print(
        alternative_logits_processor(
            (1, 2, 3, 4),
            torch.rand(100),
            n_codebooks=4,
            per_codebook_size=20,
            offset=10
        )
    )

    print(
        alternative_logits_processor(
            (),
            torch.rand(100),
            n_codebooks=4,
            per_codebook_size=20,
            offset=10
        )
    )
