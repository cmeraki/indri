from common import TEXT, SEMANTIC
from common import Config as cfg
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

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
    ds = ['Isotonic/human_assistant_conversation',
          'hakurei/open-instruct-v1']
    
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

        sample[HUMAN] = elem['instruction'].lower()
        sample[ASSISTANT] = elem['output'].lower()
        sample[ID] = idx
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


def instruct_to_semantic():
    # add kv caching to gpt to make this faster
    from tts.infer import AudioSemantic
    tokenizer = AudioSemantic()
    human_token = tokenizer.text_tokenizer.encode(HUMAN)
    assistant_token = tokenizer.text_tokenizer.encode(ASSISTANT)
    print('human_token', human_token, 'assistant_token', assistant_token)
    output_dir = 'instruct_tokens/'
    arr = []

    total = 0
    good = 0

    for sample in tqdm(iter_open_instruct()):
        human = sample[HUMAN]
        assistant = sample[ASSISTANT]
        id = sample[ID]
        if (len(human) > 100) or (len(assistant) > 100):
            continue
        
        try:
            human_tokens = [to_text_tokens(human, tokenizer),
                            to_semantic_tokens(human, tokenizer)]
            
            assistant_tokens = [to_text_tokens(assistant, tokenizer), 
                                to_semantic_tokens(assistant, tokenizer)]

            sid = 0
            for i in human_tokens:
                for j in assistant_tokens:
                    alltokens = np.hstack([human_token, i, assistant_token, j])

                    opath = output_dir + f'{id}_{sid}.npy'
                    np.save(opath, alltokens)
                    sid += 1
        
        except:
            print('failed', id, human, assistant)

    print(good, total)

if __name__ == '__main__':
     instruct_to_semantic()