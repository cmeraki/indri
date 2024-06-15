import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, Wav2Vec2BertProcessor, SeamlessM4TFeatureExtractor
import torch
from datasets import load_dataset
torch.set_default_device('cuda:0')

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
model.encoder.layers = model.encoder.layers[:7]

print(model)

samples = [dataset[0]["audio"]["array"] for i in range(128)]
samples = np.asarray(samples)

inputs = processor(samples, sampling_rate=sampling_rate, return_tensors="pt")
print(inputs.input_features.shape)
print(dataset[0]["audio"]["array"].shape)

from tqdm import tqdm
with torch.no_grad():
    for i in tqdm(range(1000)):
        outputs = model(**inputs)
#
# print(outputs.last_hidden_state.shape)
# print(outputs.extract_features.shape)