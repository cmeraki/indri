import re
import torch
import numpy as np
from typing import Tuple

from .logger import get_logger
logger = get_logger(__name__)

def deserialize_tokens(tokens):
    cb1 = tokens[0::4]
    cb2 = tokens[1::4]
    cb3 = tokens[2::4]
    cb4 = tokens[3::4]

    min_shape = min(cb1.shape, cb2.shape, cb3.shape, cb4.shape)[0]
    acoustic_tokens = np.stack([cb1[:min_shape], cb2[:min_shape] - 2048, cb3[:min_shape] - 4096, cb4[:min_shape] - 6144])

    assert acoustic_tokens.shape == (4, min_shape), 'Deserialized tokens does not have the correct shape'
    return acoustic_tokens


def sanitize_text(text: str, max_context_words: int) -> list[str]:
    """
    Sanitize text to be used for TTS

    Args:
        text (str): Text to sanitize
        max_context_words (int): Maximum number of words in a sentence

    Returns:
        list[str]: List of sentences, split by punctuation (., !, ?)
    """
    text = text.lower()

    # Remove more than one newlines and tabs
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove non-alphanumeric characters except for , . ? !
    # allowed_pattern = r'[^a-z0-9\s,\.?\n\!]'
    # text = re.sub(allowed_pattern, '', text)

    # Remove more than one punctuation mark
    text = re.sub(r'([,\.?])+', r'\1', text)

    # Split sentences by max context length
    total_words = text.split(' ')
    sentences = []
    current_sentence = ''

    for i in range(0, len(total_words), max_context_words):
        current_sentence = ' '.join(total_words[i:i+max_context_words])
        sentences.append(current_sentence.strip())

    return sentences

    # Split sentences by punctuation (., !, ?)
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
    codebook_indices = len(past_token_ids) % num_codebooks

    start_idx = offset + codebook_indices * codebook_size
    end_idx = offset + (codebook_indices + 1) * codebook_size

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

