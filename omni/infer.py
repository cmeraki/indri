import sys
sys.path.append('..')

import torch
import numpy as np

from commons import DEVICE, CTX,  TEXT, MIMI, CONVERT
from commons import Config as cfg
from omni.hfload import convert_to_hf

from train_with_mimi import get_text_tokenizer
from transformers import MimiModel, AutoFeatureExtractor
from transformers import LogitsProcessor

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


class Infer:
    def __init__(self, model_path):
        self.model = MimiModel.from_pretrained("kyutai/mimi")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

        self.omni_model = convert_to_hf(model_path,
                                    device='cuda:0')

        self.text_tokenizer = get_text_tokenizer()

        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)

        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])
        self.acoustic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[MIMI])
    
    def deserialize_tokens(self, tokens, num_codebooks):
        cb = [tokens[i::num_codebooks] for i in range(num_codebooks)]
        min_shape = min([c.shape for c in cb])[0]
        acoustic_tokens = np.stack([c[:min_shape] - 2048 * i for i, c in enumerate(cb)])
        return acoustic_tokens

    def normalize_text(self, text):
        text = text.lower()
        text = text.replace("<comma>", ',')
        text = text.replace("<period>", '.')
        text = text.replace('<questionmark>', '?')
        text = text.replace('<exclamationpoint>', '!')
        text = text.replace("\n", " ")
        return text

    @torch.inference_mode()
    def audio_infer(self, audio_tokens):
        input_tokens = np.hstack([
            self.acoustic_modality_token,
            audio_tokens + cfg.OFFSET[MIMI]
        ])
        
        input_tokens = (torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...])
        with CTX:
            self.omni_model.generation_config.eos_token_id = self.stop_token
            semantic_tokens = self.omni_model.generate(
                input_tokens,
                max_length=1024,
                temperature=0.9,
                top_k=50,
                do_sample=True,
                logits_processor=[AlternatingCodebooksLogitsProcessor(input_start_len=len(input_tokens[0]),
                                                                    codebook_size=2048,
                                                                    num_codebooks=4,
                                                                    offset=cfg.OFFSET[MIMI],
                                                                    stop_token=self.stop_token)]
            )
            semantic_tokens = semantic_tokens.detach().cpu().numpy()
            
            sem_tokens = semantic_tokens[0][1:]
            last = np.where(sem_tokens==self.stop_token)[0]
            if last.any():
                full_semantic_tokens = sem_tokens[:last[0]] - cfg.OFFSET[MIMI]
            else:
                full_semantic_tokens = sem_tokens - cfg.OFFSET[MIMI]
        full_semantic_tokens = full_semantic_tokens[:(len(full_semantic_tokens)//4)*4]
        # full_semantic_tokens = np.hstack(full_semantic_tokens)
        mimi_tokens = self.deserialize_tokens(full_semantic_tokens)

        out = self.model.decode(torch.tensor(np.expand_dims(mimi_tokens, axis=0)))
        return out.audio_values
    
    @torch.inference_mode()
    def infer(self, text, speaker='[spkr_jenny_jenny]'):
        sentences = text.split('\n')
        full_semantic_tokens = []
        
        print(sentences)

        for text in sentences:
            text = self.normalize_text(text=text)
            txt_toks = self.text_tokenizer.encode(text)
            speaker_id = self.text_tokenizer.encode(speaker)

            input_tokens = np.hstack([
                self.text_modality_token,
                txt_toks,
                self.convert_token,
                self.acoustic_modality_token,
                speaker_id,
            ])
            input_tokens = (torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...])
            
            with CTX:
                self.omni_model.generation_config.eos_token_id = self.stop_token
                semantic_tokens = self.omni_model.generate(
                    input_tokens,
                    max_length=1024,
                    temperature=0.5,
                    top_k=15,
                    do_sample=True,
                    logits_processor=[AlternatingCodebooksLogitsProcessor(input_start_len=len(input_tokens[0]),
                                                                        codebook_size=2048,
                                                                        num_codebooks=4,
                                                                        offset=cfg.OFFSET[MIMI],
                                                                        stop_token=self.stop_token)]
                )
                semantic_tokens = semantic_tokens.detach().cpu().numpy()
                
            sem_tokens = semantic_tokens[0][len(input_tokens[0]):]
            last = np.where(sem_tokens==self.stop_token)[0][0]
            sem_tokens = sem_tokens[:last] - cfg.OFFSET[MIMI]
            full_semantic_tokens.append(sem_tokens)

        full_semantic_tokens = np.hstack(full_semantic_tokens)
        mimi_tokens = self.deserialize_tokens(full_semantic_tokens)

        out = self.model.decode(torch.tensor(np.expand_dims(mimi_tokens, axis=0)))
        return out.audio_values
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()    
    args = parser.parse_args()

    model = Infer('/home/.cache/indri/models/mimi_all/gpt_last.pt')
    audio = model.infer("""Democracy is one of the most revered and widely embraced systems of governance.""")
    from silero_vad import save_audio
    print(audio[0].shape)
    save_audio('test.wav', audio[0][0], 24000)



