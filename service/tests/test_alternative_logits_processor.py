import pytest
import torch

from ..utils import alternative_logits_processor

@pytest.fixture
def mock_cfg():
    class MockCfg:
        OFFSET = {'mimi': 1000}
    return MockCfg()

@pytest.fixture
def mock_logger(mocker):
    return mocker.patch('service.tts.logger')

def test_alternative_logits_processor_shape(mock_cfg, mock_logger, monkeypatch):
    monkeypatch.setattr('service.tts.cfg', mock_cfg)

    vocab_size = 10000
    past_token_ids = (1, 2, 3)
    logits = torch.rand(vocab_size)

    result = alternative_logits_processor(past_token_ids, logits)

    assert result.shape == logits.shape
    mock_logger.info.assert_any_call(f'Logits shape: {logits.shape}, past_token_ids: {len(past_token_ids)}')
    mock_logger.info.assert_any_call('Codebook indices: 3')

@pytest.mark.parametrize("past_token_ids, expected_codebook", [
    (tuple(), 0),
    ((1,), 1),
    ((1, 2), 2),
    ((1, 2, 3), 3),
    ((1, 2, 3, 4), 0),
])
def test_alternative_logits_processor_codebook_selection(mock_cfg, mock_logger, monkeypatch, past_token_ids, expected_codebook):
    monkeypatch.setattr('service.tts.cfg', mock_cfg)
    
    vocab_size = 10000
    logits = torch.rand(vocab_size)
    
    result = alternative_logits_processor(past_token_ids, logits)
    
    # Check if only the expected codebook is non-zero
    for i in range(4):
        start = mock_cfg.OFFSET[MIMI] + i * 2048
        end = mock_cfg.OFFSET[MIMI] + (i + 1) * 2048
        if i == expected_codebook:
            assert torch.all(result[start:end] != 0)
        else:
            assert torch.all(result[start:end] == 0)

def test_alternative_logits_processor_mask(mock_cfg, mock_logger, monkeypatch):
    monkeypatch.setattr('service.tts.cfg', mock_cfg)
    
    vocab_size = 10000
    past_token_ids = (1, 2)
    logits = torch.ones(vocab_size)

    result = alternative_logits_processor(past_token_ids, logits)

    # Check if the mask is correctly applied
    assert torch.all(result[:mock_cfg.OFFSET[MIMI]] == 0)
    assert torch.all(result[mock_cfg.OFFSET[MIMI] + 4 * 2048:] == 0)
    assert torch.all(result[mock_cfg.OFFSET[MIMI] + 2 * 2048:mock_cfg.OFFSET[MIMI] + 3 * 2048] == 1)

def test_alternative_logits_processor_preservation(mock_cfg, mock_logger, monkeypatch):
    monkeypatch.setattr('service.tts.cfg', mock_cfg)
    
    vocab_size = 10000
    past_token_ids = (1,)
    logits = torch.rand(vocab_size)
    original_logits = logits.clone()
    
    result = alternative_logits_processor(past_token_ids, logits)
    
    # Check if the function preserves the original logits in the selected codebook
    start = mock_cfg.OFFSET[MIMI] + 1 * 2048
    end = mock_cfg.OFFSET[MIMI] + 2 * 2048
    assert torch.all(result[start:end] == original_logits[start:end])

if __name__ == "__main__":
    """
    Run the tests:
    pytest -v test_alternative_logits_processor.py
    """
    pytest.main()