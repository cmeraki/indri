from gpt2_multimodal_surgery import GPT
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('mdouglas/llmc-gpt2-124M-400B')
model = GPT.from_pretrained('mdouglas/llmc-gpt2-124M-400B')

model.expand_vocab(new_vocab_size=60000)
print(model)

tokens = tokenizer.encode("Capital of france is", return_tensors='pt')
for i in range(100):
    out = model.generate(tokens,
                         max_new_tokens=20,
                         temperature=0.5)



    print(out[0])
    print(tokenizer.decode(out[0]))