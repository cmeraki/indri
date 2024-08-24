import re
import math
import torch
import numpy as np
from huggingface_hub import snapshot_download

from encodec.utils import save_audio

from common import cache_dir
from common import Config as cfg
from common import SEMANTIC, TEXT, ACOUSTIC, device, ctx
from datalib.tokenlib import get_tokenizer
from tts.gpt2_model import get_model
from tts.utils import read_audio_file

def load_model(path):
    print(f'Loading model from {path}')
    model = get_model(
        vocab_size=cfg.VOCAB_SIZE,
        device=device,
        compile=True,
        path=path
    )

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

def generate_long(
        model,
        source,
        target,
        source_tokens,
        device,
        **generate_kwargs
    ):
 
    prompt_dict = generate_kwargs.get("prompt_dict")
    temperature = generate_kwargs.get("temperature", 0.9)
    top_k = generate_kwargs.get("top_k", 100)
    max_source_tokens = generate_kwargs.get("max_source_tokens", 256)
    source_overlap = generate_kwargs.get("source_overlap", 128)
    max_new_tokens = generate_kwargs.get("max_new_tokens", 3072)

    all_source_toks = []
    all_gen_toks = []

    source_tokens = source_tokens + cfg.OFFSET[source]

    if prompt_dict:
        prompt_source_tokens = prompt_dict.get('source_tokens') + cfg.OFFSET[source]
        prompt_target_tokens = prompt_dict.get('target_tokens') + cfg.OFFSET[target]

        print(f'Prompt source tokens: {prompt_source_tokens.shape}, prompt target tokens: {prompt_target_tokens.shape}')

    source_overlap = source_overlap
    target_overlap = 0
    source_stride = max_source_tokens - source_overlap

    # Initialize as empty
    target_tokens = np.asarray([])

    print(
        f'Source tokens shape: {source_tokens.shape}, Overlap: {source_overlap}, stride: {source_stride}, max tokens: {max_source_tokens}\n'
    )

    idx = 0
    while idx < source_tokens.shape[-1]:
        end_idx = idx + max_source_tokens
        source_cut = source_tokens[idx: end_idx]
        target_cut = target_tokens[-target_overlap:]

        if idx == 0 and prompt_dict:
            # end_idx = max_source_tokens - prompt_source_tokens.shape[-1]
            # source_cut = source_tokens[idx: end_idx]
            input_tokens = np.hstack([
                prompt_source_tokens,
                source_cut,
                cfg.INFER_TOKEN[target],
                prompt_target_tokens
            ])

        else:
            input_tokens = np.hstack([
                source_cut,
                cfg.INFER_TOKEN[target],
                target_cut
            ])

        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=device)[None, ...]

        print(f'{idx}: Target cut shape: {target_cut.shape}, overlap: {target_overlap}')
        print(f'{idx}: Source tokens shape: {source_cut.shape}, {input_tokens.shape}, start idx: {idx}, end idx: {end_idx}')

        with torch.no_grad():
            with ctx:
                new_target_tokens = model.generate(
                    input_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    stop_token=cfg.STOP_TOKEN[target]
                ).detach().cpu().numpy()[0]
                print(f'{idx}: Total gen shape: {new_target_tokens.shape}')

        # Only take newly generated tokens
        new_target_tokens = new_target_tokens[input_tokens.shape[-1]:]

        all_source_toks.append(input_tokens)
        all_gen_toks.append(new_target_tokens)

        # There are a few checks that we run to make sure the generation is correct
        # 1. The generated tokens are even
        # 2. The generated tokens are from the correct codebook
        # 3. The generated tokens are not too long

        if target == ACOUSTIC and new_target_tokens.shape[-1] % 2 != 0:
            print(f'Target tokens shape: {new_target_tokens.shape} is not even')
            return target_tokens, all_source_toks, all_gen_toks

        # Update the target overlap ratio, for x toks, we generate y toks
        num_source_new_toks = end_idx-idx
        if idx:
            num_source_new_toks -= source_overlap
        target_overlap = source_overlap * new_target_tokens.shape[-1]/num_source_new_toks
        target_overlap = math.ceil(target_overlap)
        target_overlap = target_overlap + 1 if target_overlap%2 != 0 else target_overlap

        print(f'{idx}: X toks: {num_source_new_toks}, Y toks: {new_target_tokens.shape}, overlap: {target_overlap}')
        # Merge into existing target tokens
        target_tokens = np.hstack([target_tokens, new_target_tokens])
        print(f'{idx}: Overall target shape is now: {target_tokens.shape}')

        print('\n')

        if end_idx > source_tokens.shape[-1]:
            break

        # if idx == 0 and prompt_dict:
        #     idx = end_idx
        # else:
        idx += source_stride

    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens, all_source_toks, all_gen_toks

class AudioSemantic:
    def __init__(self, size='125m'):
        # snapshot_download(f'cmeraki/tts_xl_30k_long_125m_en', local_dir=model_dir)
        # snapshot_download(f'cmeraki/tts_en_xl_{size}', local_dir=model_dir)

        model_dir = f'{cache_dir}/models/tts_xl_30k_long_125m_en/'
        self.text_semantic_model = load_model(path=f'{model_dir}/text_semantic/gpt_last.pt')

        model_dir = f'{cache_dir}/models/tts_en_xl_{size}/'
        self.semantic_acoustic_model = load_model(path=f'{model_dir}/semantic_acoustic/gpt_last.pt')

        self.text_tokenizer = get_tokenizer(TEXT, device=device)
        self.acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=device)

    def text_to_semantic_long(self, text, **generate_kwargs):
        """
        Convert text to semantic tokens
        Split text by <period> and tokenize each sentence
        Generate semantic tokens for each sentence
        Return concatenated semantic tokens
        """
        text = normalize_text(text).split(" <period>")[:-1]
        sentences = [(r + " <period>").strip() for r in text]

        semantic_tokens = []
        for sentence in sentences:
            sem_toks, _, _ = generate_long(
                model=self.text_semantic_model,
                source=TEXT,
                target=SEMANTIC,
                source_tokens=np.array(self.text_tokenizer.encode(sentence)),
                device=device,
                **generate_kwargs
            )
            semantic_tokens.extend(sem_toks)

        return np.array(semantic_tokens).astype(np.int64)


    def semantic_to_audio_long(self, tokens, **generate_kwargs):
        acoustic_tokens = generate_long(
            model=self.semantic_acoustic_model,
            source=SEMANTIC,
            target=ACOUSTIC,
            source_tokens=tokens,
            device=device,
            **generate_kwargs
        )

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav


    def audio_to_semantic(self, waveform=None, wav=None):
        if wav:
            waveform = read_audio_file(wav)

        acoustic_tokens = self.audio_to_semantic.encode(waveform)
        return acoustic_tokens

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s.,]', '', text)

    text = text.replace('.', ' <period>')
    text = text.replace(',', ' <comma>')
    text = text.replace('\n', ' ')

    return text.strip()


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
