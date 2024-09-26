import math
import json
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from enum import Enum
from typing import List
from openai import OpenAI
from pydantic import BaseModel
from encodec.utils import save_audio
from huggingface_hub import snapshot_download

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
    path=Path('~/Downloads/omni_text_sem_good_readin_large_indians.pt').expanduser(),
    device=DEVICE
)
semantic_acoustic_model = convert_to_hf(
    path=Path('~/Downloads/sem_aco_57k_911.pt').expanduser(),
    device=DEVICE
)
omni_model.eval()
semantic_acoustic_model.eval()

acoustic_tokenizer = get_tokenizer(ACOUSTIC, device=DEVICE)
text_tokenizer = get_text_tokenizer()
llm_client = OpenAI()

dl = TaskGenerator(loader=None, full_batches=True)
omni_model.generation_config.eos_token_id = dl.stop_token
semantic_acoustic_model.generation_config.eos_token_id = dl.stop_token

SYS_PROMPT = """
Create a short conversation between a Narrator and a Listener on the topic provided by the user. The discussion should:

1. Use easy words that 5-10 year olds can understand
2. Have short turns for each speaker
3. Include 3-6 back-and-forth exchanges
4. Be educational but fun, focusing on the given topic

Guidelines:

1. Keep sentences short and use simple language throughout the conversation
2. The Narrator should explain things clearly and simply
4. The Listener should ask curious questions a child might have
5. Avoid complex terminology

Remember to adapt the complexity of the explanation to suit a 5-10 year old audience, regardless of the topic provided.
"""

# Classes required for structured response from LLM
class Speaker(Enum):
    NARRATOR = 'Narrator'
    LISTENER = 'Listener'

class Dialogue(BaseModel):
    speaker: Speaker
    text: str

class Discussion(BaseModel):
    dialogue: List[Dialogue]

# Manual mapping of allowed speaker IDs in the app
NARRATOR = {
    'Jenny': '[spkr_jenny_jenny]',
    'Asmr': '[spkr_youtube_en_asmr_daily_bread_asmr]',
    'Attenborough': '[spkr_audiobooks_attenborough_attenborough]'
}

LISTENER = {
    'Jenny': '[spkr_jenny_jenny]',
    'Asmr': '[spkr_youtube_en_asmr_daily_bread_asmr]',
    'Attenborough': '[spkr_audiobooks_attenborough_attenborough]'
}

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


def llm(topic):
    completion = llm_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": SYS_PROMPT.strip()},
            {"role": "user", "content": f"Topic: {topic}. Remember to use short sentences in every turn."},
        ],
        response_format=Discussion,
    )

    discussion: Discussion = completion.choices[0].message.parsed

    return discussion.dialogue


def tts(text, speaker, prompt_audio):

    prompt_tokens = {}

    text_sem_output = text_sem(text, '[spkr_hifi_tts_9017]')
    sem_aco_output = sem_aco(text_sem_output, speaker, prompt_audio)
    aco_audio_output = aco_audio(sem_aco_output)

    prompt_tokens[SEMANTIC] = text_sem_output
    prompt_tokens[ACOUSTIC] = sem_aco_output

    return aco_audio_output, prompt_tokens


def generate_discussion(topic, narrator, listener):
    narrator = NARRATOR[narrator]
    listener = LISTENER[listener]

    discussion = llm(topic)
    discussion_audio = np.array([])

    narrator_prompt_audio = {}
    listener_prompt_audio = {}

    for dialogue in discussion:
        if dialogue.speaker == Speaker.NARRATOR:
            (_, audio_out), narrator_prompt_audio = tts(dialogue.text, narrator, narrator_prompt_audio)

        else:
            (_, audio_out), listener_prompt_audio = tts(dialogue.text, listener, listener_prompt_audio)
        
        narrator_prompt_audio = {}
        listener_prompt_audio = {}


        # Add artificial silence to the audio output
        discussion_audio = np.hstack([discussion_audio, np.pad(audio_out, (0, np.random.randint(1, 6000)))])

    discussion_txt = [t.text for t in discussion]
    discussion_txt = '\n'.join(discussion_txt)

    return (24000, discussion_audio), discussion_txt


with gr.Blocks() as demo:
    gr.Markdown("## Listen to a Story")

    with gr.Row():
        with gr.Column():
            narrator = gr.Dropdown(list(NARRATOR.keys()), label="Select Storyteller", value='Attenborough')
            listener = gr.Dropdown(list(LISTENER.keys()), label="Select Listener", value='Jenny')
            topic = gr.Textbox(label="Topic")

            generate_button = gr.Button("New discussion")
            story_output = gr.Textbox(label="Story")
            audio_output = gr.Audio(label="Discussion", streaming=True)

    generate_button.click(
        fn=generate_discussion,
        inputs=[topic, narrator, listener],
        outputs=[audio_output, story_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
