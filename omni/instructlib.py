from common import TEXT, SEMANTIC
from common import Config as cfg
from common import cache_dir
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import re
from pathlib import Path
import time

# tts datasets (audio-semantic)
# <text> how do i say this : 'text' </text> <assistant> semantic_tokens </assitant>
# how do i say this in english ?
# what is the word that comes after ?
# can you transcribe this text for me ?
# ----------------------

# instruct datasets (text-semantic)
# human <text tokens>
# assistant <semantic tokens>

# Using tts causes problems because there is not enough 
# variation in the dataset
# spirit used mixture of text-semantic tokens by alternating them
# it can be approached similarily by doing either side of a conversation
# in semantic  

HUMAN = 'human'
ASSISTANT = 'bot'
ID = 'id'

def split_on_period(text):
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    return re.split(pattern, text)


def iter_tiny_stories():
    dataset = load_dataset('roneneldan/TinyStories')
    elems = [(idx, elem) for (idx,elem) in tqdm(enumerate(dataset['train']), desc='loading in mem..')]
    for idx, elem in elems:
        yield idx, elem


def to_text_tokens(text, tokenizer):
    tokens = np.asarray(tokenizer.text_tokenizer.encode(text))
    tokens = tokens + cfg.OFFSET[TEXT]
    tokens = np.hstack([cfg.INFER_TOKEN[TEXT],
                        tokens,
                        cfg.STOP_TOKEN[TEXT]])
    return tokens


def to_semantic_tokens(text, tokenizer):
    tokens = tokenizer.text_to_semantic(text)
    tokens = tokens + cfg.OFFSET[SEMANTIC]
    tokens = np.hstack([cfg.INFER_TOKEN[SEMANTIC],
                        tokens,
                        cfg.STOP_TOKEN[SEMANTIC]])
    return tokens


def normalize_text(text):
    text = text.lower()
    text = text.replace(",", " <comma>")
    text = text.replace(".", " <period>")
    text = text.replace('?', ' <questionmark>')
    text = text.replace("!", '<exclamationpoint>')
    text = text.replace("\n", " ")
    return text


def split_into_sentences(text):
    allsplits = []
    for split in text.split('\n'):
        moresplits = split_on_period(split)
        allsplits.extend(moresplits)

    allsplits_sent = []
    regex = re.compile('[^a-z ]')
    for split in allsplits:
        if split:
            split = split.strip()
            split = split.replace('.', '')
            split = split.replace('"', '')
            split = split.lower()
            split = regex.sub('', split)
            if len(split) > 0:
                allsplits_sent.append(split)

    return allsplits_sent

def batch_list(input_list, batch_size=32):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

def make_stories_dataset():
    from tts.infer import AudioSemantic
    import json
    
    tokenizer = AudioSemantic(size='125m')
    output_dir = Path(f'{cache_dir}/tinystories_omni/')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    large_batch_size = 2560
    batch_size = 256
    
    large_batch = []
    story_mapping = []
    stories_raw = {}

    for story_index, sample in tqdm(iter_tiny_stories(), 'preparing samples:'):
        text = sample['text']
        sentences = split_into_sentences(text)
        stories_raw[story_index] = sentences
        
        sentences = [tokenizer.text_tokenizer.encode(s) for s in sentences]
        large_batch.extend(sentences)
        story_mapping.extend([story_index]*len(sentences))

        if len(large_batch) >= large_batch_size:
            large_batch = list(enumerate(large_batch))
            sorted_batch = sorted(large_batch, key=lambda x:len(x[1]))
            batches = batch_list(sorted_batch, batch_size=batch_size)
            results = []
            for batch in batches:
                token_batch = [i[1] for i in batch]
                idx_batch = [i[0] for i in batch]
                audio_tokens = tokenizer.text_to_semantic_batch(text_tokens=token_batch)
                audio_tokens = [i.tolist() for i in audio_tokens]
                # audio_tokens = token_batch
                batch_results = list(zip(idx_batch, audio_tokens))
                results.extend(batch_results)
            
            sorted_results = sorted(results, key=lambda x:x[0], reverse=False)
            
            stories_text = {}
            stories_audio = {}
            for residx, idx in enumerate(story_mapping):
                stories_text[idx] = stories_text.get(idx, [])
                stories_text[idx].append(large_batch[residx][1])

                stories_audio[idx] = stories_audio.get(idx, [])
                stories_audio[idx].append(sorted_results[residx][1])
            
            for key in stories_text:
                fd = {TEXT: stories_text[key],
                      SEMANTIC: stories_audio[key],
                      'raw': stories_raw[key]}
                
                fdj = json.dumps(fd)
                with open(output_dir / f'{key}.json', 'w') as writer:
                    writer.write(fdj)

            large_batch = []
            story_mapping = []

if __name__ == '__main__':
    # instruct_to_semantic()
    make_stories_dataset()