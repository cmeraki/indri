import torch
import faiss

import numpy as np

from huggingface_hub import hf_hub_download
from encodec import EncodecModel
from transformers import HubertModel, Wav2Vec2FeatureExtractor

import joblib
from audio_utils import pad_batch
import bark
from itertools import groupby

SEMANTIC = 'semantic'
ACOUSTIC = 'acoustic'


class HubertTokenizer:
    def __init__(self, device='cpu'):
        self.type = SEMANTIC
        self.vocab_size = 1000
        self.token_sample_rate = 50
        self.audio_sample_rate = 16000

        self.device = device

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("voidful/mhubert-base")
        self.hubert_model = HubertModel.from_pretrained("voidful/mhubert-base")

        kmeans_path = hf_hub_download(repo_id='voidful/mhubert-base', filename='mhubert_base_vp_en_es_fr_it3_L11_km1000.bin')

        self.output_layer = 11

        self.hubert_model.to(device)
        self.hubert_model = torch.compile(self.hubert_model)

        self.km = joblib.load(kmeans_path)
        self.C_np = self.km.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

        # self.km.cluster_centers_ = self.km.cluster_centers_.astype(np.float64)
        print("HuBert ready to tokenize")

    def encode(self, waveforms: list):
        """
        Create embeddings with Hubert model
        Classify embeddings into one of the pre-prepared 1000 clusters
        https://github.com/voidful/asrp/blob/main/asrp/voice2code.py
        """
        waveforms = self.processor(waveforms[0], sampling_rate=self.audio_sample_rate, return_tensors='pt').input_values
        waveforms = waveforms.to(self.device)
        embeddings = self.hubert_model.forward(waveforms, output_hidden_states=True).hidden_states
        embeddings = embeddings[self.output_layer].squeeze()

        dist = torch.sqrt(
            embeddings.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(embeddings, self.C)
            + self.Cnorm
        )

        min_dist = torch.topk(dist.detach(), 6, dim=-1, largest=False)
        greedy_output = min_dist.indices.T.cpu().numpy()[0]
        return greedy_output

    def decode(self):
        raise NotImplementedError


class EncodecTokenizer:
    def __init__(self, device='cpu', n_codebooks=2):
        self.type = ACOUSTIC

        self.audio_sample_rate = 24000
        self.token_sample_rate = 75
        self.n_codebooks = n_codebooks
        self.per_codebook_size = 1024
        self.vocab_size = self.n_codebooks * self.per_codebook_size

        self.output_bandwidth = ((self.token_sample_rate * # 75Hz
                                 self.n_codebooks * # 2 codebooks
                                 np.log2(self.per_codebook_size)) # 10bit per token
                                 / 1000)  # 1.5kbps

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

    def encode(self, waveform):
        """
        Encodec returns n_codebooks per token
        Here we flatten and return them as separate tokens
        Decode will decode this stream back to audio

        waveforms is a list of mono audio arrays.
        Multichannel is not supported
        """

        # padded_waveforms, sizes = pad_batch(waveforms)
        waveform = torch.unsqueeze(waveform, 1)
        waveform = waveform.to(self.device)
        encoded_frames = self.model.encode(waveform)

        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
        codes = codes.detach()[0].cpu()
        # codes = self.codebook_encoding(codes)
        # codes = self.add_start_token(codes)
        return codes

    def deserialize_tokens(self, tokens):
        cb1 = tokens[::2]
        cb2 = tokens[1::2]
        acoustic_tokens = np.stack([cb1, cb2 - 1024])
        return acoustic_tokens

    def decode(self, tokens):
        model = self.load_model(bandwidth=6, device=self.device)
        tokens = self.deserialize_tokens(tokens)
        good_audio = bark.api.generate_fine(x_coarse_gen=tokens[0:2, :], silent=False)
        good_audio = np.expand_dims(good_audio, axis=0)
        good_audio = torch.from_numpy(good_audio)
        wav = model.decode([(good_audio, None)])
        wav = wav.detach()
        return wav

    @staticmethod
    def codebook_encoding(arr: torch.tensor, per_codebook_size: int):
        c, n = arr.shape
        i_values = np.arange(c) * per_codebook_size
        arr += i_values.reshape(c, 1)
        flat_arr = arr.t().contiguous().view(c * n)
        return flat_arr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Encode audio files.')
    parser.add_argument('--audio', type=str, required=True, help='Input directory for audio files.')
    args = parser.parse_args()

    # tokenizer = EncodecTokenizer()

    tokenizer = HubertTokenizer()

    from tqdm import tqdm
    from audio_utils import read_audio_file, save_audio

    waveform = read_audio_file(args.audio, sample_rate=tokenizer.audio_sample_rate)
    waveform = waveform[0:1, :]

    print(waveform.shape)

    tokens = tokenizer.encode(waveform)
    print(tokens.shape)


    # for i in tqdm(range(1000)):
    #     tokens = tokenizer.encode(waveform)
    #     print(tokens.shape)


    # waveform = tokenizer.decode(tokens)[0]
    # print(waveform)
    # save_audio(wav=waveform,
    #            path='test.wav',
    #            sample_rate=tokenizer.audio_sample_rate)