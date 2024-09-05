import json
import torch
import random
import numpy as np
from pathlib import Path

from datalib.tokenlib import get_tokenizer
from omni.gpt2_trainer import train as gpt_train
from omni.gpt2_model import GPT
from omni.logger import get_logger

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR
from configs.constants import *
import configs.training_omni as training_cfg

logger = get_logger(__name__)
logger.info(cfg.__dict__)

def replace_consecutive(arr):
    mask = np.concatenate(([True], arr[1:] != arr[:-1]))
    return arr[mask]


# Datasets
# TTS speaker datasets : hifi, cmu, expresso, jenny + mlseng
# TTS datasets : gs, ps, mlseng
# TTS cleanup required in text : gigaspeech
# mlseng need to recreate metadata
# Cartoon data scraping : reject data from onepiece, need to scrape a good set
# prepare annotations on gigaspeech, it was made long long ago

class DataLoader:
    def __init__(self, interleaved_dirs, datasets_dirs, speaker_files):
        # types of data on which training is done
        # TTS
        # TTS with speaker id
        # ASR
        # continuation
            # text-text
            # text-semantic
            # semantic-text
        # raw semantic
        # raw text

        self.interleaved_dirs = interleaved_dirs
        self.dataset_dirs = datasets_dirs
        self.speaker_files = speaker_files

        self.text_tokenizer = get_tokenizer(type=TEXT, device='cpu')

        for idx in range(cfg.VOCAB_SIZES[SEMANTIC]):
            self.text_tokenizer.tokenizer.add_tokens(f'[sem_{idx}]')
            # print(f'Added token [sem_{idx}], {self.text_tokenizer.encode(f"[sem_{idx}]")}')

        for idx in range(cfg.VOCAB_SIZES[ACOUSTIC]):
            self.text_tokenizer.tokenizer.add_tokens(f'[aco_{idx}]')
            # print(f'Added token [aco_{idx}], {self.text_tokenizer.encode(f"[aco_{idx}]")}')

        for tok in list(cfg.MODALITY_TOKENS.values()) + list(cfg.TASK_TOKENS.values()) + [cfg.STOP_TOKEN]:
            self.text_tokenizer.tokenizer.add_tokens(tok)
            logger.info(f'Added token {tok}, {self.text_tokenizer.encode(tok)}')

        self.load_parallel_data(self.dataset_dirs)
        self.interleaved_files = self.load_interleaved(self.interleaved_dirs)
        allowed_speakers = self.load_speakers(self.speaker_files)

        self.valid_speakers = []
        # Create special tokens for each speaker
        for speaker in allowed_speakers:
            ds_name = speaker['dataset']
            speaker_id = speaker['speaker']
            combined = f'[spkr_{ds_name}_{speaker_id}]'
            self.text_tokenizer.tokenizer.add_tokens(combined)
            self.valid_speakers.append(combined)
            logger.info(f'Added token {combined}, {self.text_tokenizer.encode(combined)}')

        self.text_tokenizer.tokenizer.add_tokens(f'[spkr_unk]')
        self.convert_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONVERT])
        self.continue_token = self.text_tokenizer.encode(cfg.TASK_TOKENS[CONTINUE])
        self.stop_token = self.text_tokenizer.encode(cfg.STOP_TOKEN)
        self.semantic_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[SEMANTIC])
        self.text_modality_token = self.text_tokenizer.encode(cfg.MODALITY_TOKENS[TEXT])

    def load_speakers(self, speaker_files):
        allowed_speakers = []

        for fl in speaker_files:
            for line in open(fl):
                if not line:
                    continue
                x = json.loads(line.strip())
                allowed_speakers.append(x)

        return allowed_speakers

    def speaker_id_to_text(self, id):
        if id is None:
            return self.text_tokenizer.encode('[spkr_unk]')

        dataset = self.metadata[id]['dataset']
        speaker_id = self.metadata[id]['speaker_id']

        if speaker_id is None:
            return self.text_tokenizer.encode('[spkr_unk]')

        text = f'[spkr_{dataset}_{speaker_id}]'

        if text in self.valid_speakers:
            return self.text_tokenizer.encode(text)

        return self.text_tokenizer.encode('[spkr_unk]')

    def load_parallel_data(self, dirs):
        metadata = {}
        for dir in dirs:
            dir = Path(dir)
            metadata_path =  dir / 'annotation' / 'metadata.jsonl'
            for line in open(metadata_path):
                _metadata = json.loads(line.strip())
                _metadata['dir'] = dir
                _metadata['dataset'] = dir.name
                metadata[_metadata['id']] = _metadata

        logger.info(f"num metadata lines: {len(metadata)}")

        self.ids = list(metadata.keys())
        
        self.ids = {
            'train' : self.ids[1000:],
            'val' : self.ids[:1000],
        }

        self.metadata = metadata

    def load_interleaved(self, data_dir: list[Path]):

        files = []
        for dir in data_dir:
            files.extend(list(dir.glob('**/*.json')))

        logger.info(f"Num interleaved files: {len(files)}")

        files = {
            'train': files[1000:],
            'val': files[:1000]
        }

        return files

    def get_tokens_continue(self, split):
        filename = random.choice(self.interleaved_files[split])
        data = json.load(open(filename))
        size = len(data[TEXT])
        modalities = [TEXT, SEMANTIC]
        choices = [random.randint(0, 1) for _ in range(size)]

        output = []
        # for idx in range(size):
        #     choice = choices[idx]
        #     modality = modalities[choice]
        #     tokens = np.asarray(data[modality][idx])
        #     tokens = decorate(tokens, modality)
        #     output.extend(tokens)

        for idx in range(size):
            choice = choices[idx]
            modality = modalities[choice]
            tokens = np.asarray(data[modality][idx])
            tokens = tokens + cfg.OFFSET[modality]
            if idx == 0 and modality == TEXT:
                tokens = np.hstack([self.text_modality_token, tokens])
            elif idx == 0 and modality == SEMANTIC:
                tokens = np.hstack([self.semantic_modality_token, tokens])
            elif modality == TEXT:
                tokens = np.hstack([self.continue_token, self.text_modality_token, tokens])
            elif modality == SEMANTIC:
                tokens = np.hstack([self.continue_token, self.semantic_modality_token, tokens])
            output.extend(tokens)

        output = np.hstack([output, self.stop_token])
        return output

    def normalize_text(self, text):
        text = text.lower()
        text = text.replace("<comma>", ',')
        text = text.replace("<period>", '.')
        text = text.replace('<questionmark>', '?')
        text = text.replace('<exclamationpoint>', '!')
        text = text.replace("\n", " ")
        return text

    def get_text_tokens_for_id(self, id):
        sample = self.metadata[id]
        text = sample['raw_text']
        norm_text = self.normalize_text(text)
        tokens = np.asarray(self.text_tokenizer.encode(norm_text))
        return tokens

    def get_tokens_text(self, split):
        id = random.choice(self.ids[split])
        tokens = self.get_text_tokens_for_id(id)
        tokens = tokens + cfg.OFFSET[TEXT]

        tokens = np.hstack([
            self.text_modality_token,
            tokens,
            self.stop_token
        ])
        return tokens

    def get_tokens_speech(self, split):
        id = random.choice(self.ids[split])
        filename = self.get_tokens_path(id, SEMANTIC)
        tokens = np.load(filename)
        tokens = tokens.reshape(-1)
        tokens = replace_consecutive(tokens)
        tokens = tokens + cfg.OFFSET[SEMANTIC]

        speaker_tokens = self.speaker_id_to_text(id)

        tokens = np.hstack([
            self.semantic_modality_token,
            speaker_tokens,
            tokens,
            self.stop_token
        ])
        return tokens

    def get_tokens_path(self, id, type):
        sample = self.metadata[id]
        return sample['dir'] / sample[f'{type}_tokens'].replace('.opus', '').replace('.flac', '')

    def get_tts(self, split):
        # changing the way loading happens for asr and tts
        # now that we have speaker ids, loading would start from metadata
        id = random.choice(self.ids[split])
        text_tokens = self.get_text_tokens_for_id(id) + cfg.OFFSET[TEXT]
        speech_tokens = np.load(self.get_tokens_path(id, SEMANTIC)).reshape(-1) + cfg.OFFSET[SEMANTIC]
        speech_tokens = replace_consecutive(speech_tokens)

        speaker_tokens = self.speaker_id_to_text(id)

        tokens = np.hstack([
            self.text_modality_token,
            text_tokens,
            self.convert_token,
            self.semantic_modality_token,
            speaker_tokens,
            speech_tokens,
            self.stop_token
        ])

        return tokens
    
    def get_asr(self, split):
        # changing the way loading happens for asr and tts
        # now that we have speaker ids, loading would start from metadata
        id = random.choice(self.ids[split])
        text_tokens = self.get_text_tokens_for_id(id) + cfg.OFFSET[TEXT]
        speech_tokens = np.load(self.get_tokens_path(id, SEMANTIC)).reshape(-1) + cfg.OFFSET[SEMANTIC]
        speech_tokens = replace_consecutive(speech_tokens)

        speaker_tokens = self.speaker_id_to_text(id)

        tokens = np.hstack([
            self.semantic_modality_token,
            speaker_tokens,
            speech_tokens,
            self.convert_token,
            self.text_modality_token,
            text_tokens,
            self.stop_token
        ])
        return tokens

    def get_valid_sequence(self, split):
        methods = [
            self.get_tokens_speech,
            self.get_tokens_text,
            self.get_asr,
            self.get_tts,
            # self.get_tokens_continue,
        ]

        while True:
            try:
                method = random.choice(methods)
                # print(method)
                yield method(split), method.__name__
            except Exception as e:
                pass

    def load_batch(self, split, block_size, batch_size):
        x = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        y = np.zeros(shape=(batch_size, block_size), dtype=np.int64)
        tasks = []

        # prepopulate with pad tokens
        # so we don't have to pad later
        x = x + self.stop_token
        y = y + self.stop_token

        i = 0
        start_idx = 0
        end_idx = 0

        for tokens, task_name in self.get_valid_sequence(split):
            new_toks = block_size - end_idx
            tasks.append(task_name)
            _x = tokens[:new_toks]
            _y = tokens[1:new_toks + 1]

            if len(_x) != len(_y):
                _y = np.hstack([_y, self.stop_token])

            end_idx = start_idx + len(_x)

            x[i][start_idx:end_idx] = _x
            y[i][start_idx:end_idx] = _y

            start_idx = end_idx

            if end_idx < block_size:
                continue

            i += 1
            start_idx = 0
            end_idx = 0

            if i == batch_size:
                break

        # import pdb; pdb.set_trace()
        return x, y, tasks

    def get_batch(self, split, device, block_size, batch_size):
        x, y, tasks = self.load_batch(split,
                               block_size=block_size,
                               batch_size=batch_size)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if 'cuda' in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y, tasks


def train_omni(args):
    out_dir = Path(f'{CACHE_DIR}/models/omni_tasks_large_full_sprk/')

    interleaved_dir = [Path(f'{CACHE_DIR}/tinystories_omni/')]

    datasets_dirs = [
        Path(f'{CACHE_DIR}/jenny/'),
        Path(f'{CACHE_DIR}/expresso/'),
        Path(f'{CACHE_DIR}/mls_eng_10k/'),
        Path(f'{CACHE_DIR}/data/peoples_speech_tokens/'),
        Path(f'{CACHE_DIR}/data/gs_xl_en_tokens/'),
        Path(f'{CACHE_DIR}/hifi_tts'),
        Path(f'{CACHE_DIR}/hifi_tts_other'),
        Path(f'{CACHE_DIR}/youtube_en_gibiasmr'),
        Path(f'{CACHE_DIR}/youtube_en_asmr'),
        Path(f'{CACHE_DIR}/youtube_en_spongebob'),
    ]

    speaker_files = [
        Path(f'allowed_speakers.jsonl'),
    ]

    data_generator = DataLoader(
        interleaved_dirs=interleaved_dir,
        datasets_dirs=datasets_dirs,
        speaker_files=speaker_files,
    )

    pretrained = 'mdouglas/llmc-gpt2-774M-150B'
    vocab_size = cfg.VOCAB_SIZE

    model = GPT.from_pretrained(model_type=pretrained)
    model.expand_vocab(new_vocab_size=vocab_size)
    model.to(DEVICE)

    model = torch.compile(model)

    logger.info(model)

    logger.info(f"Vocab size: {vocab_size}")
    logger.info(f"Model outdir: {out_dir}")
    logger.info("Training omni".upper())


    gpt_train(model,
              get_batch=data_generator.get_batch,
              out_dir=out_dir,
              steps=training_cfg.STEPS,
              block_size=training_cfg.BLOCK_SIZE,
              eval_interval=training_cfg.EVAL_INTERVAL,
              batch_size=training_cfg.BATCH_SIZE,
              grad_accum_steps=training_cfg.GRAD_ACCUM_STEPS,
              device=DEVICE)

    return out_dir


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default=DEVICE)
    args = parser.parse_args()
    DEVICE = args.device

    train_omni(args)