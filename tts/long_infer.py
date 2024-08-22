import numpy as np
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

from encodec.utils import save_audio

from tts.gpt2_model import get_model
from tts.train import DataLoader
from common import SEMANTIC, TEXT, ACOUSTIC, device, ctx
from common import Config as cfg
from datalib.tokenlib import get_tokenizer
from common import cache_dir

from common import Config as cfg
from tts.utils import read_audio_file

def preprocess_text(text):
    text = text.lower()
    text = text.replace(",", " <comma>")
    text = text.replace(".", " <period>")
    text = text.replace("\n"," ")
    return text

def load_model(path):
    print(path)
    model = get_model(vocab_size=cfg.VOCAB_SIZE,
                      device=device, 
                      compile=True,
                      path=path)

    model.eval()
    return model

def extract_new_tokens(y, target):
    start_idx = np.where(y == cfg.INFER_TOKEN[target])[0]
    end_idx = np.where(y == cfg.STOP_TOKEN[target])[0]
    if end_idx.any():
        y = y[start_idx[0] + 1: end_idx[0]]
    else:
        y = y[start_idx[0] + 1:]

    return y

def generate(model, source, target, source_tokens):
    source_tokens = DataLoader.prepare_source(source_tokens,
                                            source=source,
                                            max_source_tokens=cfg.max_source_tokens)
    
    source_tokens = np.hstack([source_tokens, cfg.INFER_TOKEN[target]])
    input_tokens = (torch.tensor(source_tokens,
                                dtype=torch.long,
                                device=device)[None, ...])
    
    
    with torch.no_grad():
        with ctx:
            target_tokens = model.generate(input_tokens,
                                1024,
                                temperature=0.8,
                                top_k=100,
                                stop_token=cfg.STOP_TOKEN[target])
            
            target_tokens = target_tokens.detach().cpu().numpy()[0]
    
    target_tokens = extract_new_tokens(target_tokens, target=target)
        
    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens


def generate_long(model, source, target, source_tokens):
    source_tokens = source_tokens + cfg.OFFSET[source]
    max_tokens_one_shot = cfg.max_source_tokens//2
    source_context_size = 64
    target_context_size = source_context_size * 3
    
    target_tokens = np.asarray([])
    source_index = 0
    print(source_tokens.shape)
    while source_index < len(source_tokens):
        source_cut = source_tokens[source_index: source_index + max_tokens_one_shot]
        source_index = source_index + max_tokens_one_shot - source_context_size
        target_cut = target_tokens[-target_context_size:]
        input_tokens = np.hstack([source_cut,
                            cfg.INFER_TOKEN[target], 
                            target_cut])

        
        input_tokens = (torch.tensor(input_tokens,
                                dtype=torch.long,
                                device=device)[None, ...])
        
        with torch.no_grad():
            with ctx:
                new_target_tokens = model.generate(input_tokens,
                                    1024,
                                    temperature=0.8,
                                    top_k=100,
                                    stop_token=cfg.STOP_TOKEN[target])
                
                new_target_tokens = new_target_tokens.detach().cpu().numpy()[0]
        
        new_target_tokens = new_target_tokens[len(input_tokens[0]):]
        target_tokens = np.hstack([target_tokens, new_target_tokens])
    
    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens

class AudioSemantic:
    def __init__(self, size='125m'):
        model_dir = f'{cache_dir}/models/tts_en_xl_{size}/'
        snapshot_download(f'cmeraki/tts_en_xl_{size}', local_dir=model_dir)

        self.text_semantic_model = load_model(path=f'{model_dir}/text_semantic/gpt_last.pt')
        self.semantic_acoustic_model = load_model(path=f'{model_dir}/semantic_acoustic/gpt_last.pt')
        self.text_tokenizer = get_tokenizer(TEXT, device='cpu')
        self.acoustic_tokenizer = get_tokenizer(ACOUSTIC, device='cpu')
        self.semantic_tokenizer = get_tokenizer(SEMANTIC, device=device)

    def text_to_semantic(self, text):
        text_tokens = np.asarray(self.text_tokenizer.encode(text))
        semantic_tokens = generate(model=self.text_semantic_model,
                                   source_tokens=text_tokens,
                                   source=TEXT,
                                   target=SEMANTIC)
        return semantic_tokens

    def text_to_semantic_long(self, text):
        """
        Convert text to semantic tokens
        Split text by <period> and tokenize each sentence
        Generate semantic tokens for each sentence
        Return concatenated semantic tokens
        """
        text = preprocess_text(text)
        # Split text by <period> and tokenize each sentence
        sentences = text.split('<period>')

        semantic_tokens = []
        for sentence in sentences:
            sentence_tokens = np.asarray(self.text_tokenizer.encode(sentence))
            semantic_tokens.extend(
                generate(
                    model=self.text_semantic_model,
                    source_tokens=sentence_tokens,
                    source=TEXT,
                    target=SEMANTIC
                )
            )

        return np.array(semantic_tokens)

    def semantic_to_audio(self, tokens):
        acoustic_tokens = generate(model=self.semantic_acoustic_model,
                                   source_tokens=tokens,
                                   source=SEMANTIC,
                                   target=ACOUSTIC)

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav

    def semantic_to_audio_long(self, tokens):
        acoustic_tokens = generate_long(model=self.semantic_acoustic_model, 
                                source_tokens=tokens,
                                source=SEMANTIC,
                                target=ACOUSTIC)

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav


    def audio_to_semantic(self, waveform=None, wav=None):
        if wav:
            waveform = read_audio_file(wav)

        acoustic_tokens = self.audio_to_semantic.encode(waveform)
        return acoustic_tokens

def normalize_text(text):
    text = text.lower()
    text = text.replace(",", " <comma>")
    text = text.replace(".", " <period>")
    text = text.replace("\n"," ")
    return text

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--size', default='125m', required=False)
    parser.add_argument('--text', default='this is a test <comma> one you should not fail <period>', required=False)
    parser.add_argument('--output', default='test.wav', required=False)
    
    args = parser.parse_args()
    
    # this story has 500 semantic tokens
    text = "There was a young boy in a village. He watched the sheep for the villagers. One day, he got bored. He shouted, wolf wolf. The villagers came running to help."

    text = normalize_text(text)
    
    semlib = AudioSemantic(size=args.size)
    for i in range(100):
        semantic_tokens = semlib.text_to_semantic(text)

        
        wav = semlib.semantic_to_audio(semantic_tokens)
        print("=============")
        print("Writing output to", args.output)
        save_audio(wav=wav[0], path=f'test_{i}.wav', sample_rate=24000)
        print("=============")
