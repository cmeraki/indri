import torch

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

from tqdm import tqdm
from gpt2_multimodal_surgery import GPT
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('mdouglas/llmc-gpt2-124M-400B')
model = GPT.from_pretrained('mdouglas/llmc-gpt2-124M-400B')

model.expand_vocab(new_vocab_size=60000)
model = model.to('cuda:0')

model = torch.compile(model)

tokens = tokenizer.encode("Capital of france is", return_tensors='pt')

tokens = tokens.to('cuda:0')
with torch.inference_mode():
    with  torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in tqdm(range(100)):
            out = model.generate(tokens,
                                 max_new_tokens=200,
                                 temperature=0.5)


    # print(out[0])
    # print(tokenizer.decode(out[0]))