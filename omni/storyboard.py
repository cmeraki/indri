import gradio as gr
import numpy as np
from enum import Enum
from typing import List
from openai import OpenAI
from pydantic import BaseModel
from encodec.utils import save_audio

from logger import get_logger
from infer import Infer
from tqdm import tqdm

DEVICE = 'cuda:0'

logger = get_logger(__name__)
llm_client = OpenAI()

model_infer = Infer(model_path='/home/meraki/Downloads/mimi_speaker_ids_249k.pt')

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
    'Mark': '[spkr_hifi_tts_9017]',
}

LISTENER = {
    'Jenny': '[spkr_jenny_jenny]',
    'Mark': '[spkr_hifi_tts_9017]',
}

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


def tts(text, speaker):
    wav = model_infer.infer(text, speaker)
    wav = wav[0][0].detach()
    wav = wav.cpu()

    return 24_000, wav.numpy()


def generate_discussion(topic, narrator, listener):
    narrator = NARRATOR[narrator]
    listener = LISTENER[listener]

    discussion = llm(topic)
    discussion_audio = np.array([])

    for dialogue in tqdm(discussion):
        if dialogue.speaker == Speaker.NARRATOR:
            _, audio_out = tts(dialogue.text, narrator)
        else:
            _, audio_out = tts(dialogue.text, listener)
        
        discussion_audio = np.hstack([discussion_audio, np.pad(audio_out, (0, np.random.randint(1, 6000)))])

    discussion_txt = [t.text for t in discussion]
    discussion_txt = '\n'.join(discussion_txt)

    return (24000, discussion_audio), discussion_txt


with gr.Blocks() as demo:
    gr.Markdown("## Listen to a Story")

    with gr.Row():
        with gr.Column():
            narrator = gr.Dropdown(list(NARRATOR.keys()), label="Select Storyteller", value='Mark')
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
