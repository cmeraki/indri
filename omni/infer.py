from transformers import LogitsProcessor
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
        scores_processed[:, stop_token] = scores[:, stop_token]
        return scores_processed