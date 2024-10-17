import re
import torch
import numpy as np
from typing import Tuple

from transformers import MimiModel, AutoFeatureExtractor
from transformers import LogitsProcessor

from .logger import get_logger

logger = get_logger(__name__)

DEVICE = 'cuda:0'

class AlternatingCodebooksLogitsProcessor(LogitsProcessor):
    def __init__(self, input_start_len: int, codebook_size: int, num_codebooks: int, offset: int, stop_token: int):
        self.input_start_len = input_start_len
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.offset = offset
        self.stop_token = stop_token
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[-1]
        codebook_idx = ((curr_len - self.input_start_len) % self.num_codebooks)
        
        scores_processed = scores.clone()
        scores_processed[:, : self.offset + codebook_idx * self.codebook_size] = -float("inf")
        scores_processed[:, self.offset + (codebook_idx+1) * self.codebook_size :] = -float("inf")
        scores_processed[:, self.stop_token] = scores[:, self.stop_token]
        return scores_processed


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


def alternative_logits_processor(past_token_ids: Tuple[int], logits: torch.Tensor) -> torch.Tensor:
    """
    Logit processor for alternating codebooks
    Given a sequence of logits, we want to make sure that the alternating tokens
    are chosen from different codebooks.

    Args:
        past_token_ids: Tuple[int] - Tuple of past token ids
        logits: torch.Tensor - Logits to process. Shape (vocab_size)

    Returns:
        torch.Tensor - Processed logits. Shape (vocab_size)
    """
    kwargs = {
        'num_codebooks': 4,
        'codebook_size': 2048,
        'offset': cfg.OFFSET[MIMI],
    }

    new_logits = logits.clone()
    codebook_indices = len(past_token_ids) % kwargs['num_codebooks']

    logger.info(f'Logits shape: {logits.shape}, past_token_ids: {len(past_token_ids)}, codebook indices: {codebook_indices}')

    mask = torch.zeros_like(new_logits)
    start_idx = kwargs['offset'] + codebook_indices * kwargs['codebook_size']
    end_idx = kwargs['offset'] + (codebook_indices + 1) * kwargs['codebook_size']
    logger.info(f'Start idx: {start_idx}, end idx: {end_idx}')

    mask[start_idx:end_idx] = 1
    new_logits = new_logits * mask

    return new_logits
