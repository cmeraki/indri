from trainer import gpt_trainer
from datalib import semantic_acoustic_generator, text_semantic_generator
import tokenizer

data_dir = '../data'

dataset = GigaSpeechDataset()

tokenizer.tokenize(dataset, data_dir, type='semantic')
tokenizer.tokenize(dataset, data_dir, type='acoustic')
tokenizer.tokenize(dataset, data_dir, type='text')

t1 = gpt_trainer.create(
    layers=8,
    heads=8,
    dim=512,
)

data_generator = semantic_acoustic_generator(data_dir)

train_steps = 100_000
semantic_acoustic_model = t1.train(data_generator, train_steps, model_dir='')

t2 = gpt_trainer.create(
    layers=8,
    heads=8,
    dim=512,
)

data_generator = text_semantic_generator(data_dir)
text_acoustic_model = t2.train(data_generator, train_steps, model_dir='')

