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
        'source_tokens': toks['semantic_tokens'],
        'target_tokens': toks['acoustic_tokens']
    }

    global ts_prompt_toks_dict
    ts_prompt_toks_dict = {
        'source_tokens': toks['text_tokens'],
        'target_tokens': toks['semantic_tokens']
    }

def echo(text, speaker):
    global prev_speaker

    if not prev_speaker:
        load_prompt(speaker)
        prev_speaker = speaker

    if speaker != prev_speaker:
        load_prompt(speaker)
        prev_speaker = speaker

    sem_toks = ttslib.text_to_semantic_long(
        text,
        max_source_tokens=32,
        source_overlap=16,
        temperature=0.99,
        max_new_tokens=1024,
        prompt_dict=ts_prompt_toks_dict
    )
    print(sem_toks.shape)

    aud = vanilla_ttslib.semantic_to_audio(sem_toks)
    print(aud.shape)

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
    demo.launch()
