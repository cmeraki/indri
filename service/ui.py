import os
import json
import base64
import requests
import gradio as gr
import numpy as np
from omni.logger import get_logger

logger = get_logger(__name__)

url = 'http://localhost:8000/tts'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# Manual mapping of allowed speaker IDs in the app
SPEAKERS = []
with open('omni/allowed_speakers.jsonl', 'r') as f:
    for ln in f:
        if not f:
            continue
        SPEAKERS.append(json.loads(ln)['combined'])


def _tts(text, speaker):
    data = {
        'text': text
    }

    response = requests.post(url, headers=headers, json=data)
    data = response.json()
    decoded = base64.b64decode(data['array'])
    array = np.frombuffer(decoded, dtype=np.dtype(data['dtype'])).reshape(data['shape'])

    return 24_000, array[0][0]


with gr.Blocks() as demo:
    gr.Markdown("## Omni")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Text-to-Speech (TTS)")
            reader = gr.Dropdown(SPEAKERS, label="Select Reader")
            text_input = gr.Textbox(label="Text Input")

            audio_output = gr.Audio(label="Audio Output")
            _button = gr.Button("Generate Speech")

    _button.click(
        fn=_tts,
        inputs=[text_input, reader],
        outputs=[audio_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8001)

