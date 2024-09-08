import torch
import gradio as gr
import numpy as np
from pathlib import Path
from huggingface_hub import snapshot_download
from encodec.utils import save_audio

from configs.commons import Config as cfg
from configs.commons import DEVICE, CACHE_DIR, CTX
from configs.constants import *

from omni.hfload import convert_to_hf
from datalib.tokenlib import get_tokenizer
from omni.train_omni_instruct import DataLoader

local_dir = f'{CACHE_DIR}/models/omni_774m_tinystories'
omni_model = convert_to_hf(
    path=f'/home/.cache/indri/models/omni_tasks_large/gpt_2900.pt',
    device=DEVICE
)
semantic_acoustic_model = convert_to_hf(
    path=f'/home/.cache/indri/romit/models/semantic_acoustic_tasks_small/gpt_4500.pt',
    device=DEVICE
)

acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)

# snapshot_download(f'cmeraki/tts_xl_30k_long_125m_en/semantic_acoustic/', local_dir=local_dir)

# Common tokenization
dl = DataLoader(
    interleaved_dirs=[],
    datasets_dirs=[],
    speaker_files=[Path('../allowed_speakers.jsonl').resolve()]
)

text_tokenizer = dl.text_tokenizer
acoustic_modality_token = text_tokenizer.encode(cfg.MODALITY_TOKENS[ACOUSTIC])

omni_model.generation_config.eos_token_id = dl.stop_token
semantic_acoustic_model.generation_config.eos_token_id = dl.stop_token

SPEAKERS = {
    "ASMR": "[spkr_asmr]",
    "Cartoon": "[spkr_cartoon]",
    "Jenny": "[spkr_jenny_jenny]",
    "Random": "[spkr_unk]"
}


def create_omni_tokens(task, incoming_tokens, incoming_modality, speaker_id = '[spkr_unk]'):
    if task == TTS:
        input_tokens = np.hstack([
            dl.text_modality_token,
            incoming_tokens,
            dl.convert_tokens,
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker_id)
        ])

    elif task == ASR:
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
        input_tokens = np.hstack([
            dl.semantic_modality_token,
            text_tokenizer.encode(speaker_id),
            incoming_tokens
        ])

    print(f'Input tokens: {text_tokenizer.decode(input_tokens)}, shape: {input_tokens.shape}')
    return input_tokens


def create_semaco_tokens(incoming_tokens, speaker_id = '[spkr_unk]'):
    input_tokens = np.hstack([
        dl.semantic_modality_token,
        text_tokenizer.encode(speaker_id),
        incoming_tokens,
        dl.convert_tokens,
        acoustic_modality_token
    ])

    print(f'Input tokens: {text_tokenizer.decode(input_tokens)}, shape: {input_tokens.shape}')
    return input_tokens


def converse(task, text_input, audio_input, speaker):

    if text_input:
        assert audio_input is None, "Cannot provide both text and audio inputs"
        assert task in [TTS, CONTINUE], "Task must be TTS or Continue for text input"

        input_tokens = create_omni_tokens(task, text_input, TEXT, speaker)

    elif audio_input:
        assert text_input is None, "Cannot provide both audio and text inputs"
        assert task in [ASR, CONTINUE], "Task must be ASR or Continue for audio input"

        input_tokens = create_omni_tokens(task, audio_input, AUDIO, speaker)

    input_tokens = (torch.tensor(input_tokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Text -> Semantic/Text
    with CTX:
        omni_output = omni_model.generate(
            input_tokens,
            max_length=1024,
            temperature=0.8,
            top_k=100,
            do_sample=True
        )

        omni_output = omni_output.detach().cpu().numpy()[0]
        omni_output = omni_output[len(input_tokens):]

    end_idx = np.where(omni_output == dl.stop_token)[0][0]
    omni_output = omni_output[:end_idx]
    print(f'Omni output: {text_tokenizer.decode(omni_output)}, shape: {omni_output.shape}')

    if task in [CONTINUE, ASR] and text_input:
        return text_tokenizer.decode(omni_output), None, None

    semantic_tokens = create_semaco_tokens(omni_output)
    semantic_tokens = (torch.tensor(semantic_tokens, dtype=torch.long, device=DEVICE)[None, ...])

    # Semantic -> Acoustic
    with CTX:
        acoustic_tokens = semantic_acoustic_model.generate(
            semantic_tokens,
            max_length=3072,
            temperature=0.8,
            top_k=100,
            do_sample=True
        )
        acoustic_tokens = acoustic_tokens.detach().cpu().numpy()[0]
        acoustic_tokens = acoustic_tokens[len(semantic_tokens):]

    end_idx = np.where(acoustic_tokens == dl.stop_token)[0][0]
    acoustic_tokens = acoustic_tokens[0:end_idx]
    acoustic_tokens = acoustic_tokens - cfg.OFFSET[ACOUSTIC]

    if len(acoustic_tokens) % 2 == 1:
        acoustic_tokens = acoustic_tokens[:-1]

    print(f'Acoustic output: {acoustic_tokens}, shape: {acoustic_tokens.shape}')

    # Acoustic -> Audio
    wav = acoustic_tokenizer.decode(torch.tensor(acoustic_tokens))
    wav = wav[0].cpu()

    tmp_audio_file = f'omni.wav'
    save_audio(wav, tmp_audio_file, sample_rate=24000)

    return None, 24_000, wav[0].numpy()

with gr.Blocks() as demo:
    gr.Markdown("## Omni")
    
    with gr.Row():
        task = gr.Dropdown(["ASR", "TTS", "Continue Text", "Continue Audio"], label="Select Task")
        speaker = gr.Dropdown(list(SPEAKERS.keys()), label="Select Speaker")

    with gr.Row():
        text_input = gr.Textbox(label="Text Input")
        audio_input = gr.Audio(sources=["microphone", "upload"], label="Audio Input", type="numpy")
    
    with gr.Row():
        text_output = gr.Textbox(label="Text Output")
        audio_output = gr.Audio(label="Audio Output")

    submit_button = gr.Button("Submit")

    submit_button.click(
        fn=converse,
        inputs=[task, text_input, audio_input, SPEAKERS[speaker]],
        outputs=[text_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
