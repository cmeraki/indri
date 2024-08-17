from common import TEXT, SEMANTIC
from common import Config as cfg
from common import cache_dir
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import re

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

def load_all_datasets():
    ds = ['Isotonic/human_assistant_conversation_deduped',
          'hakurei/open-instruct-v1', 
          'SohamGhadge/casual-conversation', 
          'goendalf666/sales-conversations', 
          'jihyoung/ConversationChronicles',
          'talkmap/telecom-conversation-corpus',
          'talkmap/banking-conversation-corpus']
    
    for d in ds:
        dataset = load_dataset(d)
    
def iter_open_instruct():
    dataset = load_dataset('hakurei/open-instruct-v1')
    # ~500k rows 
    # dict_keys(['output', 'input', 'instruction'])
    # samples with input require extra cleanup
    for idx, elem in enumerate(dataset['train']):
        sample = {}
        if elem['input']:
            continue 

        sample[HUMAN] = normalize_text(elem['instruction'].lower())
        sample[ASSISTANT] = normalize_text(elem['output'].lower())
        sample[ID] = f'open_instruct_{idx}'
        yield sample

def iter_human_assistant():
    dataset = load_dataset('Isotonic/human_assistant_conversation_deduped')
    # ~500k rows 
    # dict_keys(['output', 'input', 'instruction'])
    # samples with input require extra cleanup
    for idx, elem in enumerate(dataset['train']):
        sample = {}
        sample[HUMAN] = normalize_text(elem['prompt'].lower().replace('human: ', '').replace('assistant:', '').strip())
        sample[ASSISTANT] = normalize_text(elem['response'].lower().replace('human: ', '').replace('assistant:', '').strip())
        sample[ID] = f'human_assistant_{idx}'
        yield sample

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
    text = text.replace("\n"," ")
    return text



def instruct_to_semantic():
    # add kv caching to gpt to make this faster
    from tts.infer import AudioSemantic
    tokenizer = AudioSemantic(size='30m')
    human_token = tokenizer.text_tokenizer.encode(HUMAN)
    assistant_token = tokenizer.text_tokenizer.encode(ASSISTANT)
    print('human_token', human_token, 'assistant_token', assistant_token)
    output_dir = f'{cache_dir}/instruct_tokens/'
    arr = []

    total = 0
    good = 0
    n_tokens = 0
    datasets = [iter_human_assistant, iter_open_instruct]
    samples = []
    
    code_symbols = r'[{}#$%^&*+=]'
    digits =  r'\d'
    seen = {}
    for ds in datasets:
        for sample in tqdm(ds()):
            if sample[HUMAN] and sample[ASSISTANT]:
                if (len(sample[HUMAN].split()) > 1) and (len(sample[ASSISTANT].split()) > 1):
                    if not bool(re.search(code_symbols, sample[HUMAN] + sample[ASSISTANT])):
                        if not bool(re.search(digits, sample[HUMAN] + sample[ASSISTANT])):
                            x = sample[HUMAN] + sample[ASSISTANT]
                            if x not in seen:
                                seen[x] = 1
                                if (len(tokenizer.text_tokenizer.encode(sample[HUMAN])) < 100):
                                    samples.append(sample)
            # if len(samples) > 1000:
            #     break
        
    print(len(samples))
    samples = sorted(samples, key=lambda x:len(x[HUMAN]) + len(x[ASSISTANT]))
    # n_tokens = 0
    for sample in samples[:10]:
        print(sample)
    
    for sample in tqdm(samples):
        human = sample[HUMAN]
        assistant = sample[ASSISTANT]
        id = sample[ID]
        try:
            human_tokens = [to_text_tokens(human, tokenizer),   
                            to_semantic_tokens(human, tokenizer)]
            
            assistant_tokens = [to_text_tokens(assistant, tokenizer), 
                                to_semantic_tokens(assistant, tokenizer)]

            sid = 0
            for i in human_tokens:
                for j in assistant_tokens:
                    alltokens = np.hstack([human_token, i, assistant_token, j])
                    alltokens = alltokens.astype(dtype=np.uint16)
                    # print(alltokens)
                    opath = output_dir + f'{id}_{sid}.npy'
                    np.save(opath, alltokens)
                    sid += 1
        
        except:
            print('failed', id, human, assistant)

    print(good, total, n_tokens)

if __name__ == '__main__':
     instruct_to_semantic()