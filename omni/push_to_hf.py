"""
Push the model and tokenizer to Hugging Face
"""

import os
import sys

sys.path.append('omni/')

from .hfload import convert_to_hf
from .train_with_mimi import get_text_tokenizer

if __name__ == '__main__':
    """
    python -m omni.push_to_hf \
        --model_path /home/.cache/indri/models/mimi_all/gpt_175000.pt \
        --device cuda:0 \
        --repo_id cmeraki/mimi_tts_hf_stage
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model to push', required=True)
    parser.add_argument('--device', type=str, help='Device to use', default='cuda:0', required=False)
    parser.add_argument('--repo_id', type=str, help='The hugging face repo id to push to', required=True)
    parser.add_argument('--hf_token', type=str, help='Hugging Face token', required=False, default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--commit_message', type=str, help='Commit message', required=False, default='HF upload')

    args = parser.parse_args()

    assert args.hf_token, 'HF_TOKEN is not set, either set it in the environment or pass it as an argument (pass --hf_token)'

    print(f'Uploading model to {args.repo_id} with commit message: {args.commit_message}')

    non_hf_model = convert_to_hf(args.model_path, args.device)

    non_hf_model.push_to_hub(
        repo_id=args.repo_id,
        token=args.hf_token,
        commit_message=args.commit_message
    )

    text_tokenizer = get_text_tokenizer()
    text_tokenizer.tokenizer.push_to_hub(
        repo_id=args.repo_id,
        token=args.hf_token,
        commit_message=args.commit_message
    )

    print('Done!')
