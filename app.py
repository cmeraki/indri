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

import pdb

if gr.NO_RELOAD:
    logger = get_logger(__name__)

    # local_dir = f'{CACHE_DIR}/models/omni_774m_tinystories'
    # snapshot_download(f'cmeraki/tts_xl_30k_long_125m_en/semantic_acoustic/', local_dir=local_dir)

    omni_model = convert_to_hf(
        path=f'omni_less_steps.pt',
        device=DEVICE
    )
    semantic_acoustic_model = convert_to_hf(
        path=f'sem_aco.pt',
        device=DEVICE
    )

    acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)
    text_tokenizer = get_text_tokenizer()
    semantic_tokenizer = AudioToken(tokenizer=Tokenizers.semantic_s, device=DEVICE)

    dl = TaskGenerator(loader=None)
    omni_model.generation_config.eos_token_id = dl.stop_token
    semantic_acoustic_model.generation_config.eos_token_id = dl.stop_token

    # Manual mapping of allowed speaker IDs in the app
    SPEAKERS = {
        "ASMR": "[spkr_youtube_en_gibiasmr_gibi_asmr]",
        "Cartoon": "[spkr_youtube_en_spongebob_spongebob]",
        "Jenny": "[spkr_jenny_jenny]",
        "Random": cfg.UNKNOWN_SPEAKER_ID
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


def create_omni_tokens(task, incoming_tokens, incoming_modality, speaker_id = '[spkr_unk]'):
    logger.info(f'Incoming tokens: {incoming_tokens}, shape: {incoming_tokens.shape}, modality: {incoming_modality}')

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

    logger.info(f'OMNI INPUT TOKENS: {text_tokenizer.decode(input_tokens)}, shape: {input_tokens.shape}')
    return input_tokens


def create_semaco_tokens(incoming_tokens, speaker_id = '[spkr_unk]'):
    input_tokens = np.hstack([
        dl.semantic_modality_token,
        incoming_tokens,
        dl.convert_token,
        dl.acoustic_modality_token,
        text_tokenizer.encode(speaker_id)
    ])

    logger.info(f'SEMACO INPUT TOKENS: {text_tokenizer.decode(input_tokens)}, shape: {input_tokens.shape}')
    return input_tokens


def _tts(text, speaker):
    logger.info(f'Text: {text}, Speaker: {speaker}')

    speaker = SPEAKERS[speaker]
    text = normalize_text(text)
    text_tokens = text_tokenizer.encode(text)
    input_tokens = create_omni_tokens(TTS, np.array(text_tokens), TEXT, speaker)

    input_tokens = (torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Text -> Semantic
    with CTX:
        omni_output = omni_model.generate(
            input_tokens,
            max_length=1024,
            temperature=0.8,
            top_k=100,
            do_sample=True
        )

        omni_output = omni_output.detach().cpu().numpy()[0]
        omni_output = omni_output[input_tokens.shape[-1]:]

    end_idx = np.where(omni_output == dl.stop_token)[0]
    if len(end_idx) > 1:
        end_idx = end_idx[0]
        omni_output = omni_output[:end_idx]

    logger.info(f'OMNI OUTPUT: {text_tokenizer.decode(omni_output)}, shape: {omni_output.shape}')

    semantic_tokens = create_semaco_tokens(omni_output, speaker)
    semantic_tokens = (torch.tensor(semantic_tokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Semantic -> Acoustic
    with CTX:
        acoustic_tokens = semantic_acoustic_model.generate(
            semantic_tokens,
            max_length=3072,
            temperature=0.9,
            top_k=100,
            do_sample=True
        )
        acoustic_tokens = acoustic_tokens.detach().cpu().numpy()[0]
        acoustic_tokens = acoustic_tokens[semantic_tokens.shape[-1]:]

    end_idx = np.where(acoustic_tokens == dl.stop_token)[0]
    if len(end_idx) > 1:
        end_idx = end_idx[0]
        acoustic_tokens = acoustic_tokens[:end_idx]

    acoustic_tokens = acoustic_tokens - cfg.OFFSET[ACOUSTIC]

    # Extra check to ensure that the acoustic tokens are even
    if len(acoustic_tokens) % 2 == 1:
        acoustic_tokens = acoustic_tokens[:-1]

    logger.info(f'ACOUSTIC TOKENS: {acoustic_tokens}, shape: {acoustic_tokens.shape}')

    # Acoustic -> Audio
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    wav = wav[0].cpu()

    tmp_audio_file = f'omni.wav'
    save_audio(wav, tmp_audio_file, sample_rate=24000)

    return 24_000, wav[0].numpy()


def _asr(input, speaker):
    speaker = SPEAKERS[speaker]

    sr, y = input
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y = y.reshape(1, -1)
    y = torch.tensor(y)

    aud = convert_audio(y, sr, target_sr=16_000, target_channels=1)
    aud = transform_func(aud)

    logger.info(f'Audio shape: {aud.shape}')

    audio_tokens = semantic_tokenizer.encode(aud).numpy()[0][0]
    logger.info(f'Audio tokens shape: {audio_tokens.shape}')
    audio_tokens = replace_consecutive(audio_tokens)
    input_tokens = create_omni_tokens(ASR, audio_tokens, AUDIO, speaker)

    input_tokens = (torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Semantic -> Text
    with CTX:
        omni_output = omni_model.generate(
            input_tokens,
            max_length=1024,
            temperature=0.8,
            top_k=100,
            do_sample=True
        )

        omni_output = omni_output.detach().cpu().numpy()[0]
        omni_output = omni_output[input_tokens.shape[-1]:]

    end_idx = np.where(omni_output == dl.stop_token)[0]
    if len(end_idx) > 1:
        end_idx = end_idx[0]
        omni_output = omni_output[:end_idx]

    logger.info(f'OMNI OUTPUT: {text_tokenizer.decode(omni_output)}, shape: {omni_output.shape}')

    return text_tokenizer.decode(omni_output)


with gr.Blocks() as demo:
    gr.Markdown("## Omni")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Text-to-Speech (TTS)")
            speaker_tts = gr.Dropdown(list(SPEAKERS.keys()), label="Select Speaker")
            text_input = gr.Textbox(label="Text Input")
            audio_output = gr.Audio(label="Audio Output")
            tts_button = gr.Button("Generate Speech")

        with gr.Column():
            gr.Markdown("### Speech-to-Text (ASR)")
            speaker_asr = gr.Dropdown(list(SPEAKERS.keys()), label="Select Speaker")
            audio_input = gr.Audio(sources=["microphone", "upload"], label="Audio Input", type="numpy")
            text_output = gr.Textbox(label="Text Output")
            asr_button = gr.Button("Generate Text")

    tts_button.click(
        fn=_tts,
        inputs=[text_input, speaker_tts],
        outputs=[audio_output]
    )

    asr_button.click(
        fn=_asr,
        inputs=[audio_input, speaker_asr],
        outputs=[text_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
    # _tts(text='how are you', speaker='Jenny')
