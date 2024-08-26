import gradio as gr
import numpy as np
from audiotoken import AudioToken, Tokenizers

from tts.long_infer import AudioSemantic
from tts.infer import AudioSemantic as VanillaAudioSemantic

ttslib = AudioSemantic()
vanilla_ttslib = VanillaAudioSemantic()

acoustic_tokenizer = AudioToken(Tokenizers.acoustic, device='cuda:0')
semantic_tokenizer = AudioToken(Tokenizers.semantic_s, device='cuda:0')
prev_speaker = None
sa_prompt_toks_dict = None
ts_prompt_toks_dict = None

def load_prompt(speaker):
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

    sem_toks = ttslib.text_to_semantic_long(
        text,
        max_source_tokens=32,
        source_overlap=16,
        temperature=0.99,
        max_new_tokens=1024,
        prompt_dict=ts_prompt_toks_dict
    )
    print(f'Semantic tokens shape: {sem_toks.shape}')

    print(f'Generating audio tokens {sem_toks.shape}')

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
