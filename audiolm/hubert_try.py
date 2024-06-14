# Load model directly
import numpy as np
import torch
from transformers import AutoProcessor, AutoModel, HubertModel, Wav2Vec2Processor
import torchaudio
from encodec.utils import convert_audio
from torch.nn import functional as F

x, sr = torchaudio.load('/home/apurva/projects/indri/data/audio/0028_portuguese_nonscripted_2.wav')
waveform = convert_audio(x,
                         sr,
                         target_sr=16000,
                         target_channels=1)

# waveform = F.layer_norm(waveform, waveform.shape)

# processor = AutoProcessor.from_pretrained("utter-project/mHuBERT-147")
model = AutoModel.from_pretrained("utter-project/mHuBERT-147")

# y = np.random.random(size=(1, 100, 768))

from tqdm import tqdm
import faiss
from faiss import IndexIVFFlat
index = faiss.read_index('mhubert147_faiss.index')

print(index.d, index.ntotal)
index_ivf: IndexIVFFlat = faiss.extract_index_ivf(index)


def get_centroids_index(xq, index, index_ivf: IndexIVFFlat):
    opq_mt = faiss.downcast_VectorTransform(index.chain.at(0))
    xq_t = opq_mt.apply_py(xq)
    DC, C = index_ivf.quantizer.search(xq_t, 1)
    return DC, C

# centroids = []
# for idx in tqdm(range(1000)):
#     centroid = index_ivf.quantizer.reconstruct(idx)
#     centroids.append(centroid)
#
# centroids = np.asarray(centroids)
# dot = centroids@centroids.T
# print(dot)
#
# distances, centroid_index = get_centroids_index(centroids, index, index_ivf)
#
# print(centroid_index)


for layer_idx in range(12, 1, -1):
    model.encoder.layers = model.encoder.layers[0:layer_idx]

    y = model.forward(waveform)
    y = y.last_hidden_state.detach()

    DC, C = get_centroids_index(y[0], index, index_ivf)
    print(layer_idx, DC.mean())
