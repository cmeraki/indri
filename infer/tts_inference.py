import math
import torch
import numpy as np
from pathlib import Path
from encodec.utils import save_audio
from huggingface_hub import snapshot_download

from configs.commons import Config as cfg
from configs.commons import DEVICE, CTX, CACHE_DIR
from configs.constants import *

from omni.logger import get_logger
from omni.train import get_text_tokenizer, TaskGenerator, replace_consecutive
from datalib.tokenlib import get_tokenizer

from vllm import LLM, SamplingParams

logger = get_logger(__name__)

local_dir = f'{CACHE_DIR}/models/'
snapshot_download(f'cmeraki/omni_774m_32k', repo_type='model', local_dir=Path(local_dir, 'omni_774m_32k'))
snapshot_download(f'cmeraki/sem_aco_44k', repo_type='model', local_dir=Path(local_dir, 'sem_aco_44k'))

omni_model = LLM(model='cmeraki/omni_774m_32k')
semantic_acoustic_model = LLM(model='cmeraki/sem_aco_44k')

omni_model.eval()
semantic_acoustic_model.eval()

acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)
text_tokenizer = get_text_tokenizer()

dl = TaskGenerator(loader=None, full_batches=False)


def normalize_text(text):
    text = text.lower()
    text = text.replace("<comma>", ',')
    text = text.replace("<period>", '.')
    text = text.replace('<questionmark>', '?')
    text = text.replace('<exclamationpoint>', '!')
    text = text.replace("\n", " ")
    return text


def sem_aco(semantic_tokens, speaker_id='[spkr_unk]', prompt_tokens: dict = None):
    logger.info(f'Semantic tokens shape: {semantic_tokens.shape}, speaker: {speaker_id}')

    max_source_tokens=1024  
    all_source_toks = []
    all_gen_toks = []
    target_overlap = 0
    target_tokens = np.asarray([], dtype=np.int64)
    stride = max_source_tokens//2

    for start_idx in range(0, semantic_tokens.shape[-1], stride):
        # Create proper token sequence for a short generation
        end_idx = start_idx + max_source_tokens
        source_cut = semantic_tokens[start_idx: end_idx]
        target_cut = target_tokens[-target_overlap:]

        input_tokens = create_semaco_tokens(source_cut, speaker_id, prompt_tokens)
        input_tokens = np.hstack([
            input_tokens, target_cut
        ])
        logger.info(f'{start_idx}: Source tokens shape: {source_cut.shape}, target tokens shape: {target_cut.shape}, combined shape: {input_tokens.shape}')

        input_tokens = torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        all_source_toks.append(input_tokens)

        with CTX:
            new_target_tokens = semantic_acoustic_model.generate(
                input_tokens,
                max_length=3072,
                temperature=0.8,
                top_k=100,
                do_sample=True,
            ).detach().cpu().numpy()[0]

            new_target_tokens = new_target_tokens[input_tokens.shape[-1]:]

        end_idx = np.where(new_target_tokens == dl.stop_token)[0]
        if len(end_idx) >= 1:
            end_idx = end_idx[0]
            new_target_tokens = new_target_tokens[:end_idx]

        # Extra check to ensure that the acoustic tokens are even
        if len(new_target_tokens) % 2 == 1:
            new_target_tokens = new_target_tokens[:-1]

        logger.info(f'{start_idx}: New target tokens shape: {new_target_tokens.shape}')

        target_overlap = new_target_tokens.shape[-1]
        if start_idx == 0:
            target_overlap = (max_source_tokens - stride - len(prompt_tokens.get(SEMANTIC, []))) * new_target_tokens.shape[-1]/max_source_tokens
            target_overlap = math.ceil(target_overlap)
        if target_overlap % 2 == 1:
            target_overlap += 1

        target_tokens = np.hstack([target_tokens, new_target_tokens])
        logger.info(f'{start_idx}: Overlap: {target_overlap}')
        logger.info(f'{start_idx}: SEM ACO OUTPUT SHAPE: {new_target_tokens.shape}')

        all_gen_toks.append(new_target_tokens)

    logger.info(f'SEM ACO COMPLETE OUTPUT SHAPE: {target_tokens.shape}')
    return target_tokens


def text_sem(text, speaker_id):
    logger.info(f'Text: {text}, Speaker: {speaker_id}')

    text = normalize_text(text)#.split(".")[:-1]
    #sentences = [(r + ".").strip() for r in text]
    sentences = [text]

    logger.info(f'Sentences: {sentences}')

    semantic_tokens = np.asarray([], dtype=np.int64)

    for sentence in sentences:
        text_tokens = create_omni_tokens(np.array(text_tokenizer.encode(sentence)), speaker_id)
        logger.info(f'TEXT SEM INPUT TOKENS: {text_tokenizer.decode(text_tokens)}, shape: {text_tokens.shape}')
        text_tokens = (torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)[None, ...])

        # Text -> Semantic
        with CTX:
            omni_output = omni_model.generate(
                text_tokens,
                max_length=1024,
                temperature=0.5,
                top_k=100,
                do_sample=True
            )

            omni_output = omni_output.detach().cpu().numpy()[0]
            omni_output = omni_output[text_tokens.shape[-1]:]

        end_idx = np.where(omni_output == dl.stop_token)[0]
        if len(end_idx) >= 1:
            end_idx = end_idx[0]
            omni_output = omni_output[:end_idx]

        omni_output = replace_consecutive(omni_output)
        logger.info(f'TEXT SEM OUTPUT SHAPE: {omni_output.shape}')

        semantic_tokens = np.hstack([semantic_tokens, omni_output])

    semantic_tokens = replace_consecutive(semantic_tokens)
    logger.info(f'TEXT SEM COMPLETE OUTPUT SHAPE: {semantic_tokens.shape}')

    return semantic_tokens


def aco_audio(acoustic_tokens):
    acoustic_tokens = acoustic_tokens - cfg.OFFSET[ACOUSTIC]

    # Acoustic -> Audio
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    wav = wav.cpu()

    tmp_audio_file = f'omni.wav'
    save_audio(wav, tmp_audio_file, sample_rate=24000)

    return 24_000, wav[0].numpy()


def create_omni_tokens(incoming_tokens, speaker_id):

    input_tokens = np.hstack([
        dl.text_modality_token,
        incoming_tokens,
        dl.convert_token,
        dl.semantic_modality_token,
        text_tokenizer.encode(speaker_id)
    ])

    return input_tokens


def create_semaco_tokens(incoming_tokens, speaker_id, prompt_tokens: dict = None):

    if prompt_tokens:
        input_tokens = np.hstack([
            dl.semantic_modality_token,
            prompt_tokens[SEMANTIC],
            incoming_tokens,
            dl.convert_token,
            dl.acoustic_modality_token,
            text_tokenizer.encode(speaker_id),
            prompt_tokens[ACOUSTIC],
        ])
        return input_tokens

    input_tokens = np.hstack([
        dl.semantic_modality_token,
        incoming_tokens,
        dl.convert_token,
        dl.acoustic_modality_token,
        text_tokenizer.encode(speaker_id)
    ])

    return input_tokens


def tts(text, speaker, prompt_audio):

    prompt_tokens = {}

    text_sem_output = text_sem(text, '[spkr_hifi_tts_9017]')
    sem_aco_output = sem_aco(text_sem_output, speaker, prompt_audio)
    aco_audio_output = aco_audio(sem_aco_output)

    prompt_tokens[SEMANTIC] = text_sem_output
    prompt_tokens[ACOUSTIC] = sem_aco_output

    return aco_audio_output, prompt_tokens
