from huggingface_hub import Repository
from transformers import pipeline, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY

from .tts_pipeline import IndriTTSPipeline
from src.commons import Config as cfg
from src.commons import CONVERT, CONTINUE, MIMI, TEXT

def register_pipeline(task, model_id):
    PIPELINE_REGISTRY.register_pipeline(
        task,
        pipeline_class=IndriTTSPipeline,
        pt_model=AutoModelForCausalLM
    )

    pipe = pipeline(task, model=model_id)
    return pipe


def push_to_hub(path, model_id, pipe):
    repo = Repository(path, clone_from=model_id)
    pipe.save_pretrained(path)
    repo.push_to_hub()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True, help='The model id on HF to update')
    parser.add_argument('--path', type=str, required=True, help='Local path to clone the HF repo (should be a empty directory)')

    args = parser.parse_args()

    model_id = args.model_id
    task = 'indri-tts'
    path = args.path

    pipe = register_pipeline(task, model_id)

    """
    Update some custom config for the model
    """
    pipe.tokenizer.eos_token = '[stop]'
    pipe.model.config.eos_token_id = pipe.tokenizer.encode('[stop]')[0]
    pipe.model.config.update({
        'n_ctx': 1024,
        'vocab_size': cfg.VOCAB_SIZE,
        'convert_token': cfg.TASK_TOKENS[CONVERT],
        'continue_token': cfg.TASK_TOKENS[CONTINUE],
        'text_token': cfg.MODALITY_TOKENS[TEXT],
        'mimi_token': cfg.MODALITY_TOKENS[MIMI],
        'stop_token': cfg.STOP_TOKEN,
        'num_codebooks': cfg.n_codebooks,
        'audio_offset': cfg.OFFSET[MIMI],
    })

    push_to_hub(path, model_id, pipe)
