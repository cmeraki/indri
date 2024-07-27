from huggingface_hub import snapshot_download
from os import getenv

token = getenv('HF_TOKEN')
print("Token", token)

if (not token) or token[:3]!='hf_':
    print("Token missing or incorrect. Add token as env variable HF_TOKEN")

else:
    snapshot_download('cmeraki/audio', repo_type='model', local_dir='audio_models', token=token)
    snapshot_download('cmeraki/chameleon_tokenizer', repo_type='model', local_dir='chameleon_tokenizer', token=token)
