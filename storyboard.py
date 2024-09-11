import math
import json
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from functools import partial
from huggingface_hub import snapshot_download
from encodec.utils import save_audio
from audiotoken import AudioToken, Tokenizers
from transformers import Wav2Vec2FeatureExtractor

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR, CTX
from configs.constants import *

from omni.hfload import convert_to_hf
from omni.logger import get_logger
from omni.train import get_text_tokenizer, TaskGenerator, replace_consecutive
from datalib.tokenlib import get_tokenizer

from tts.utils import convert_audio


logger = get_logger(__name__)

# local_dir = f'{CACHE_DIR}/models/'
# snapshot_download(f'cmeraki/omni_774m_32k', repo_type='model', local_dir=Path(local_dir, 'omni_774m_32k'))
# snapshot_download(f'cmeraki/sem_aco_44k', repo_type='model', local_dir=Path(local_dir, 'sem_aco_44k'))

omni_model = convert_to_hf(
    path=Path('~/Downloads/text_sem_21k_911.pt').expanduser(),
    device=DEVICE
)
semantic_acoustic_model = convert_to_hf(
    path=Path('~/Downloads/sem_aco_28k_911.pt').expanduser(),
    device=DEVICE
)
omni_model.eval()
semantic_acoustic_model.eval()

acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)
text_tokenizer = get_text_tokenizer()
semantic_tokenizer = AudioToken(tokenizer=Tokenizers.semantic_s, device=DEVICE)

dl = TaskGenerator(loader=None)
omni_model.generation_config.eos_token_id = dl.stop_token
semantic_acoustic_model.generation_config.eos_token_id = dl.stop_token

# Manual mapping of allowed speaker IDs in the app
STORYTELLERS = {
    'Jenny': '[spkr_jenny_jenny]',
    'Attenborough': '[spkr_audiobooks_attenborough_attenborough]'
}

LISTENERS = {
    'Jenny': '[spkr_jenny_jenny]',
    'Attenborough': '[spkr_audiobooks_attenborough_attenborough]'
}

def hubert_processor(audio, processor):
    return processor(
        audio,
        sampling_rate=16_000,
        return_tensors='pt'
    ).input_values[0]


processor = Wav2Vec2FeatureExtractor.from_pretrained('voidful/mhubert-base')
transform_func = partial(hubert_processor, processor=processor)

def normalize_text(text):
    text = text.lower()
    text = text.replace("<comma>", ',')
    text = text.replace("<period>", '.')
    text = text.replace('<questionmark>', '?')
    text = text.replace('<exclamationpoint>', '!')
    text = text.replace("\n", " ")
    return text


def long_infer(semantic_tokens, speaker_id='[spkr_unk]'):
    logger.info(f'Semantic tokens shape: {semantic_tokens.shape}, speaker_id: {speaker_id}')

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

        input_tokens = create_semaco_tokens(source_cut, speaker_id)
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
                temperature=0.7,
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
            target_overlap = (max_source_tokens-stride) * new_target_tokens.shape[-1]/max_source_tokens
            target_overlap = math.ceil(target_overlap)
        if target_overlap % 2 == 1:
            target_overlap += 1

        target_tokens = np.hstack([target_tokens, new_target_tokens])
        logger.info(f'{start_idx}: Target tokens shape: {target_tokens.shape}, overlap: {target_overlap}')

        all_gen_toks.append(new_target_tokens)

    return target_tokens - cfg.OFFSET[ACOUSTIC]


def split_infer(text, speaker_id):
    text = normalize_text(text).split(".")[:-1]
    sentences = [(r + ".").strip() for r in text]

    logger.info(f'Sentences: {sentences}')

    semantic_tokens = np.asarray([], dtype=np.int64)

    for sentence in sentences:
        text_tokens = create_omni_tokens(TTS, np.array(text_tokenizer.encode(sentence)), TEXT, speaker_id)
        logger.info(f'TEXT SEM INPUT TOKENS: {text_tokenizer.decode(text_tokens)}, shape: {text_tokens.shape}')
        text_tokens = (torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)[None, ...])

        # Text -> Semantic
        with CTX:
            omni_output = omni_model.generate(
                text_tokens,
                max_length=1024,
                temperature=0.7,
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
        logger.info(f'TEXT SEM OUTPUT: shape: {omni_output.shape}')

        semantic_tokens = np.hstack([semantic_tokens, omni_output])

    semantic_tokens = replace_consecutive(semantic_tokens)
    logger.info(f'TEXT SEM COMPLETE OUTPUT shape: {semantic_tokens.shape}')

    return semantic_tokens


def create_omni_tokens(task, incoming_tokens, incoming_modality, speaker_id = '[spkr_unk]'):
    logger.info(f'Tokens for text-sem, incoming tokens: {incoming_tokens}, shape: {incoming_tokens.shape}, task: {task}, modality: {incoming_modality}')

    if task == TTS:
        input_tokens = np.hstack([
            dl.text_modality_token,
            incoming_tokens,
            dl.convert_token,
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker_id)
        ])

    elif task == ASR:
        incoming_tokens = incoming_tokens + cfg.OFFSET[SEMANTIC]
        input_tokens = np.hstack([
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker_id),
            incoming_tokens,
            dl.convert_token,
            dl.text_modality_token
        ])

    elif task == CONTINUE and incoming_modality == TEXT:
        input_tokens = np.hstack([
            dl.text_modality_token,
            incoming_tokens
        ])

    elif task == CONTINUE and incoming_modality == AUDIO:
        incoming_tokens = incoming_tokens + cfg.OFFSET[SEMANTIC]
        input_tokens = np.hstack([
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker_id),
            incoming_tokens
        ])

    return input_tokens


def create_semaco_tokens(incoming_tokens, speaker_id = '[spkr_unk]'):
    input_tokens = np.hstack([
        dl.semantic_modality_token,
        incoming_tokens,
        dl.convert_token,
        dl.acoustic_modality_token,
        text_tokenizer.encode(speaker_id)
    ])

    # logger.info(f'SEM ACO INPUT TOKENS: {text_tokenizer.decode(input_tokens)}, shape: {input_tokens.shape}')
    return input_tokens


def text_sem(text, speaker):
    logger.info(f'Text: {text}, Speaker: {speaker}')

    omni_output = split_infer(text, speaker)

    logger.info(f'OMNI OUTPUT: shape: {omni_output.shape}')

    return omni_output


def sem_aco(semantic_tokens, speaker):
    acoustic_tokens = long_infer(semantic_tokens, speaker_id=speaker)

    # Acoustic -> Audio
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    wav = wav[0].cpu()

    tmp_audio_file = f'omni.wav'
    save_audio(wav, tmp_audio_file, sample_rate=24000)

    return 24_000, wav[0].numpy()


def _tts(text, speaker):

    text_sem_output = text_sem(text, speaker)
    sem_aco_output = sem_aco(text_sem_output, speaker)

    return sem_aco_output


def generate_story(topic, storyteller, listener):
    storyteller = STORYTELLERS[storyteller]
    listener = LISTENERS[listener]

    story = open('story.json').read()
    story = json.loads(story)
    story_audio = np.array([])

    for ln in story:
        _, audio_out = _tts(ln["text"], storyteller)
        story_audio = np.hstack([story_audio, audio_out])

    return 24000, story_audio


with gr.Blocks() as demo:
    gr.Markdown("## Listen to a Story")

    with gr.Row():
        with gr.Column():
            storyteller = gr.Dropdown(list(STORYTELLERS.keys()), label="Select Storyteller", value='Jenny')
            listener = gr.Dropdown(list(LISTENERS.keys()), label="Select Listener", value='Jenny')
            topic = gr.Textbox(label="Topic")

            generate_button = gr.Button("New story")
            audio_output = gr.Audio(label="Story")

    generate_button.click(
        fn=generate_story,
        inputs=[topic, storyteller, listener],
        outputs=[audio_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
    # _tts(text='how are you', speaker='Jenny')
