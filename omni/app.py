from infer import Infer
import gradio as gr
import json

model_infer = Infer(model_path='/home/meraki/Downloads/mimi_speaker_ids_249k.pt')

SPEAKERS = [json.loads(d)['combined'] for d in open('allowed_speakers.jsonl')]

def tts(text, speaker):
    wav = model_infer.infer(text, speaker)
    wav = wav[0][0].detach()
    wav = wav.cpu()

    return 24_000, wav.numpy()


with gr.Blocks() as demo:
    gr.Markdown("## Omni")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Text-to-Speech (TTS)")
            reader = gr.Dropdown(SPEAKERS, label="Select Speaker")
            text_input = gr.Textbox(label="Text Input")
            audio_output = gr.Audio(label="Audio Output")
            sem_aco_button = gr.Button("Generate Speech")

    def text_sem_wrapper(text, speaker):
        audio_output = tts(text, speaker)
        return audio_output

    sem_aco_button.click(
        fn=text_sem_wrapper,
        inputs=[text_input, reader],
        outputs=[audio_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
    # _tts(text='how are you', speaker='Jenny')
