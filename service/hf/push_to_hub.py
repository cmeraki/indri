import torch
from huggingface_hub import Repository
from transformers import pipeline, AutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY

from tts_pipeline import IndriTTSPipeline

def register_pipeline(task, model_id):
    PIPELINE_REGISTRY.register_pipeline(
        task,
        pipeline_class=IndriTTSPipeline,
        pt_model=AutoModelForCausalLM
    )

    pipe = pipeline(task, model=model_id, device=torch.device('cuda:0'))
    return pipe


def push_to_hub(path, model_id, pipe):
    repo = Repository(path, clone_from=model_id)
    pipe.save_pretrained(path)
    repo.push_to_hub()


if __name__ == "__main__":
    model_id = "cmeraki/mimi_124m_8cb"
    task = "indri-tts"
    path = "mimi_124m_8cb"

    pipe = register_pipeline(task, model_id)
    push_to_hub(path, model_id, pipe)
