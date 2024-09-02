import re
import math
import torch
import numpy as np
from huggingface_hub import snapshot_download

from encodec.utils import save_audio

from tts.gpt2_model import get_model
from tts.utils import read_audio_file
from datalib.tokenlib import get_tokenizer
from tts.config import Config as cfg, SEMANTIC, TEXT, ACOUSTIC, DEVICE, ctx, seed, cache_dir

def load_model(path):
    print(f'Loading model from {path}')
    model = get_model(
        vocab_size=cfg.VOCAB_SIZE,
        device=DEVICE,
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

def generate(model, source, target, source_tokens, device, generate_kwargs):

    temperature = generate_kwargs.get("temperature", 0.9)
    top_k = generate_kwargs.get("top_k", 100)
    max_new_tokens = generate_kwargs.get("max_new_tokens", cfg.BLOCK_SIZE[source])
    max_source_tokens = generate_kwargs.get("max_source_tokens", cfg.MAX_SOURCE_TOKENS[source])

    source_tokens = source_tokens + cfg.OFFSET[source]
    source_tokens = np.reshape(source_tokens, -1)
    source_tokens = source_tokens[0: max_source_tokens]

    source_tokens = np.hstack([source_tokens, cfg.INFER_TOKEN[target]])
    input_tokens = (torch.tensor(source_tokens, dtype=torch.long, device=device)[None, ...])

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    with torch.no_grad():
        with ctx:
            target_tokens = model.generate(
                input_tokens,
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token=cfg.STOP_TOKEN[target]
            )

            target_tokens = target_tokens.detach().cpu().numpy()[0]

    target_tokens = extract_new_tokens(target_tokens, target=target)
    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens

def generate_long(
        model,
        source,
        target,
        source_tokens,
        device,
        generate_kwargs
    ):

    prompt_dict = generate_kwargs.get("prompt_dict")
    temperature = generate_kwargs.get("temperature", 0.9)
    top_k = generate_kwargs.get("top_k", 100)
    max_source_tokens = generate_kwargs.get("max_source_tokens", cfg.MAX_SOURCE_TOKENS[source])
    source_overlap = generate_kwargs.get("source_overlap", 128)
    max_new_tokens = generate_kwargs.get("max_new_tokens", cfg.BLOCK_SIZE[source])

    print(f'Max source tokens: {max_source_tokens}, source overlap: {source_overlap}, max new tokens: {max_new_tokens}')

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

    for idx in range(0, source_tokens.shape[-1], source_stride):
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

        # print(f'{idx}: Target cut shape: {target_cut.shape}, overlap: {target_overlap}')
        print(f'{idx}: Source tokens shape: {source_cut.shape}, {input_tokens.shape}, start idx: {idx}, end idx: {end_idx}')

        # Reset the random state between generations
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

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

        if target == ACOUSTIC:
            assert (new_target_tokens[::2] - cfg.OFFSET[ACOUSTIC] > 1024).sum() == 0, 'Codebook 1 Acoustic tokens should be less than 1024'
            assert (new_target_tokens[1::2] - cfg.OFFSET[ACOUSTIC] < 1024).sum() == 0, 'Codebook 2 Acoustic tokens should be less than 1024'

        if target == ACOUSTIC and new_target_tokens.shape[-1] % 2 != 0:
            print(f'{idx}: Target tokens shape is not even, truncating last token')
            new_target_tokens = new_target_tokens[:-1]

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

    target_tokens = target_tokens - cfg.OFFSET[target]
    return target_tokens, all_source_toks, all_gen_toks

class AudioSemantic:
    def __init__(self, size='125m', custom_path: str = None):

        model_dir = f'{cache_dir}/models/tts_xl_30k_long_125m_en/'
        snapshot_download(f'cmeraki/tts_xl_30k_long_125m_en', local_dir=model_dir)
        self.text_semantic_model = load_model(path=f'{model_dir}/text_semantic/gpt_last.pt')

        model_dir = f'{cache_dir}/models/tts_en_xl_{size}/'
        snapshot_download(f'cmeraki/tts_en_xl_{size}', local_dir=model_dir)
        self.semantic_acoustic_model = load_model(path=f'{model_dir}/semantic_acoustic/gpt_last.pt')

        self.text_tokenizer = get_tokenizer(TEXT, device=DEVICE)
        self.acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)
        self.device = DEVICE

        # self.semantic_acoustic_model_new = load_model(path=f'{model_dir}/semantic_acoustic/gpt_last.pt')
        self.semantic_acoustic_model_new = load_model(path=f'{custom_path}')

    def text_to_semantic_long(self, text, generate_kwargs=None):
        """
        Convert text to semantic tokens
        Split text by <period> and tokenize each sentence
        Generate semantic tokens for each sentence
        Return concatenated semantic tokens
        """
        text = normalize_text(text).split(" <period>")[:-1]
        sentences = [(r + " <period>").strip() for r in text]

        print(f'Sentences: {sentences}')

        semantic_tokens = []
        for sentence in sentences:
            sem_toks, _, _ = generate_long(
                model=self.text_semantic_model,
                source=TEXT,
                target=SEMANTIC,
                source_tokens=np.array(self.text_tokenizer.encode(sentence)),
                device=self.device,
                generate_kwargs=generate_kwargs
            )
            semantic_tokens.extend(sem_toks)

        return np.array(semantic_tokens).astype(np.int64)


    def semantic_to_audio_long(self, tokens, retries=5, model=None, generate_kwargs=None):
        for i in range(retries):
            try:
                acoustic_tokens, _, _ = generate_long(
                    model=model,
                    source=SEMANTIC,
                    target=ACOUSTIC,
                    source_tokens=tokens,
                    device=self.device,
                    generate_kwargs=generate_kwargs
                )
                break
            except Exception as e:
                print(f'Error: {e}, retrying {i+1} of {retries}')
                continue

        if i == retries - 1:
            raise Exception('Failed to generate acoustic tokens')

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav.cpu()


    def semantic_to_audio(self, tokens, model=None, generate_kwargs=None):
        acoustic_tokens = generate(
            model=model,
            source=SEMANTIC,
            target=ACOUSTIC,
            source_tokens=tokens,
            device=self.device,
            generate_kwargs=generate_kwargs
        )

        wav = self.acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
        return wav.cpu()


    def audio_to_semantic(self, waveform=None, wav=None):
        if wav:
            waveform = read_audio_file(wav)

        acoustic_tokens = self.acoustic_tokenizer.encode(waveform)
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
    
    text = "There was a young boy in a village. He watched the sheep for the villagers."

    semlib = AudioSemantic(size=args.size)
    for i in range(10):
        semantic_tokens = semlib.text_to_semantic_long(text)

        wav = semlib.semantic_to_audio_long(semantic_tokens, model=semlib.semantic_acoustic_model)
        print("=============")
        print("Writing output to", args.output)
        save_audio(wav=wav[0], path=f'test_{i}.wav', sample_rate=24000)
        print("=============")
