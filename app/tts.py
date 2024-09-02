import gradio as gr
import numpy as np

from tts.long_infer import AudioSemantic
from tts.config import Config as cfg, TEXT, SEMANTIC, ACOUSTIC

ttslib = AudioSemantic()
prev_speaker = None

def load_prompt(speaker):
    print(f'Loading prompt for {speaker}')

    if speaker == 'jenny':
        toks = np.load('prompts/jenny_short/tokens.npz')
    elif speaker == 'lj':
        toks = np.load('prompts/lj_female_long/tokens.npz')

    global sa_prompt_toks_dict
    sa_prompt_toks_dict = {
        'source_tokens': toks['SEMANTIC'],
        'target_tokens': toks['ACOUSTIC']
    }

    global ts_prompt_toks_dict
    ts_prompt_toks_dict = {
        'source_tokens': toks['TEXT'],
        'target_tokens': toks['SEMANTIC']
    }

def echo(text, speaker):
    global prev_speaker

    if not prev_speaker:
        load_prompt(speaker)
        prev_speaker = speaker

    if speaker != prev_speaker:
        load_prompt(speaker)
        prev_speaker = speaker

    print(f'Generating semantic tokens {text}')

    generate_kwargs = {
        'max_source_tokens': 32,
        'source_overlap': 16,
        'temperature': 0.99,
        'max_new_tokens': cfg.BLOCK_SIZE[TEXT],
        'prompt_dict': ts_prompt_toks_dict
    }

    sem_toks = ttslib.text_to_semantic_long(
        text,
        generate_kwargs=generate_kwargs,
        prompt_dict=ts_prompt_toks_dict
    )
    print(f'Semantic tokens shape: {sem_toks.shape}')

    generate_kwargs = {
        'max_source_tokens': 300,
        'source_overlap': 150,
        'temperature': 0.95,
        'max_new_tokens': cfg.BLOCK_SIZE[SEMANTIC],
        'prompt_dict': sa_prompt_toks_dict
    }

    aud = ttslib.semantic_to_audio_long(
        sem_toks,
        max_source_tokens=128,
        source_overlap=64,
        temperature=0.8,
        max_new_tokens=1024,
        prompt_dict=sa_prompt_toks_dict
    )
    print(f'Audio tokens shape: {aud.shape}')

    return 24_000, aud[0][0].cpu().numpy()

demo = gr.Interface(
    fn=echo,
    inputs=[
        gr.Textbox(label="Text to convert"),
        gr.Dropdown(["jenny", "lj"], label="Speaker", value="jenny")
    ],
    outputs=gr.Audio(label="Generated Audio", autoplay=True),
    title="Text to Speech",
    description="Enter text to convert to audio"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
