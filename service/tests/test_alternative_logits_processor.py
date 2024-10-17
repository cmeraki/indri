import torch
import pytest

from ..utils import alternative_logits_processor

MIMI = 'mimi'

@pytest.fixture
def mock_cfg():
    return {
        'per_codebook_size': 2048,
        'n_codebooks': 4,
        'offset': 100,
    }

def test_alternative_logits_processor_shape(mock_cfg):

    vocab_size = 10000
    past_token_ids = (1, 2, 3)
    logits = torch.rand(vocab_size)

    result = alternative_logits_processor(past_token_ids, logits, **mock_cfg)

    assert result.shape == logits.shape

@pytest.mark.parametrize("past_token_ids, expected_codebook", [
    (tuple(), 0),
    ((1,), 1),
    ((1, 2), 2),
    ((1, 2, 3), 3),
    ((1, 2, 3, 4), 0),
])
def test_alternative_logits_processor_codebook_selection(mock_cfg, past_token_ids, expected_codebook):
    
    vocab_size = 10000
    logits = torch.rand(vocab_size)

    result = alternative_logits_processor(past_token_ids, logits, **mock_cfg)

    # Check if only the expected codebook is non-zero
    for i in range(4):
        start = mock_cfg['offset'] + i * mock_cfg['per_codebook_size']
        end = mock_cfg['offset'] + (i + 1) * mock_cfg['per_codebook_size']

        if i == expected_codebook:
            assert torch.all(result[start:end] != 0)
        else:
            assert torch.all(result[start:end] == 0)

def test_alternative_logits_processor_mask(mock_cfg):
    
    vocab_size = 10000
    past_token_ids = (1, 2)
    logits = torch.ones(vocab_size)

    result = alternative_logits_processor(past_token_ids, logits, **mock_cfg)

    # Check if the mask is correctly applied
    assert torch.all(result[:mock_cfg['offset']] == 0)
    assert torch.all(result[mock_cfg['offset'] + 4 * mock_cfg['per_codebook_size']:] == 0)
    assert torch.all(result[mock_cfg['offset'] + 2 * mock_cfg['per_codebook_size']:mock_cfg['offset'] + 3 * mock_cfg['per_codebook_size']] == 1)

def test_alternative_logits_processor_preservation(mock_cfg):
    
    vocab_size = 10000
    past_token_ids = (1,)
    logits = torch.rand(vocab_size)
    original_logits = logits.clone()
    
    result = alternative_logits_processor(past_token_ids, logits, **mock_cfg)
    
    # Check if the function preserves the original logits in the selected codebook
    start = mock_cfg['offset'] + 1 * mock_cfg['per_codebook_size']
    end = mock_cfg['offset'] + 2 * mock_cfg['per_codebook_size']
    assert torch.all(result[start:end] == original_logits[start:end])

if __name__ == "__main__":
    """
    Run the tests:
    pytest -v test_alternative_logits_processor.py
    """
    pytest.main()
