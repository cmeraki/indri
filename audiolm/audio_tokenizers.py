import torch
import torchaudio
import faiss

import numpy as np

from huggingface_hub import hf_hub_download
from encodec import EncodecModel
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from audio_utils import pad_batch
import bark

SEMANTIC = 'semantic'
ACOUSTIC = 'acoustic'

class HubertTokenizer:
    def __init__(self, pad_token=None, device='cpu'):
        self.type = SEMANTIC
        self.vocab_size = 1000 + 1 #pad token
        self.token_sample_rate = 50
        self.audio_sample_rate = 16000

        self.device = device
        self.pad_token = pad_token if pad_token else self.vocab_size

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
        self.hubert_model = HubertModel.from_pretrained("utter-project/mHuBERT-147")

        self.hubert_model.to(device)

        faiss_index_file = hf_hub_download("utter-project/mHuBERT-147",
                                           filename='mhubert147_faiss.index')

        self.index = faiss.read_index(faiss_index_file)
        self.index_ivf = faiss.extract_index_ivf(self.index)
        print("HuBert ready to tokenize")

    def encode(self, waveforms: list):
        """
        Create embeddings with Hubert model
        Classify embeddings into one of the pre-prepared 1000 clusters
        """
        # waveforms = self.processor(waveforms, sampling_rate=self.input_sr, return_tensors='pt').input_values
        embeddings = self.hubert_model.forward(waveforms)
        embeddings = embeddings.last_hidden_state.detach()[0]
        cluster_ids = self.assign_clusters(embeddings)
        return cluster_ids

    def assign_clusters(self, embeddings):
        opq_mt = faiss.downcast_VectorTransform(self.index.chain.at(0))
        xq_t = opq_mt.apply_py(embeddings)
        distances, centroid_indices = self.index_ivf.quantizer.search(xq_t, 1)
        return centroid_indices

    def decode(self):
        raise NotImplementedError


class EncodecTokenizer:
    def __init__(self, pad_token=None, device='cpu', n_codebooks=2):
        self.type = ACOUSTIC

        self.audio_sample_rate = 24000
        self.token_sample_rate = 75
        self.n_codebooks = n_codebooks
        self.per_codebook_size = 1024
        self.vocab_size = self.n_codebooks * self.per_codebook_size + 1

        self.output_bandwidth = ((self.token_sample_rate * # 75Hz
                                 self.n_codebooks * # 2 codebooks
                                 np.log2(self.per_codebook_size)) # 10bit per token
                                 / 1000)  # 1.5kbps

        self.pad_token = pad_token if pad_token else self.vocab_size

        self.device = device
        self.model = self.load_model(self.output_bandwidth, self.device)

    @staticmethod
    def load_model(bandwidth, device):
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.zero_grad()
        model.to(device)
        if 'cuda' in device:
            model = torch.compile(model)
        return model

    def encode(self, waveforms: list):
        """
        Encodec returns n_codebooks per token
        Here we flatten and return them as separate tokens
        Decode will decode this stream back to audio
        """

        padded_waveforms, sizes = pad_batch(waveforms)
        batch = padded_waveforms.to(self.device)
        encoded_frames = self.model.encode(batch)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.detach()

        expected_lengths = np.ceil(np.asarray(sizes) / (self.audio_sample_rate/self.token_sample_rate)).astype(int)

        new_codes = []
        for code, size in zip(codes, expected_lengths):
            code = code[:, :size]
            code = self.codebook_encoding(code)
            code = self.add_start_token(code)
            new_codes.append(code)

        codes = np.hstack(new_codes)
        return codes

    def decode(self):
        pass

    def deserialize_tokens(self, tokens):
        # serial token shape = n,1
        # deserialize to (codebook, tokens)
        # remove start_token
        start_indices = tokens == self.pad_token
        start_indices = np.argwhere(start_indices).reshape(-1)
        start_indices = start_indices[1:]
        splits = np.split(tokens, indices_or_sections=start_indices)
        codebook_deindex = np.arange(self.n_codebooks) * self.per_codebook_size
        codebook_deindex = np.expand_dims(codebook_deindex, axis=-1)
        splits = [split[1:len(split) - 1 + len(split) % 2].reshape((2, split[1:].shape[0] // 2),
                                                                   order='F') - codebook_deindex for split in
                  splits]
        return splits

    def decode_to_audio(self, tokens):
        model = self.load_model(bandwidth=3, device=self.device)
        tokens = self.deserialize_tokens(tokens)
        token_single = np.expand_dims(tokens[0], axis=0)
        good_audio = bark.api.generate_fine(x_coarse_gen=token_single[0, 0:2, :], silent=False)
        good_audio = np.expand_dims(good_audio, axis=0)
        good_audio = torch.from_numpy(good_audio)
        wav = model.decode([(good_audio, None)])

        return wav

    def codebook_encoding(self, arr):
        c, n = arr.shape
        i_values = np.arange(c) * self.per_codebook_size
        arr += i_values.reshape(c, 1)
        flat_arr = arr.reshape(c * n, order='F')
        return flat_arr

    def add_start_token(self, arr):
        arr = np.insert(arr, 0, self.pad_token)
        return arr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--audio', type=str, required=True, help='Input directory for audio files.')
    args = parser.parse_args()

    tokenizer = HubertTokenizer()
    # tokenizer = EncodecTokenizer()

    from audio_utils import read_audio_file
    waveform = read_audio_file(args.audio, sample_rate=tokenizer.audio_sample_rate)

    tokens = tokenizer.encode(waveform)
    print(tokens)