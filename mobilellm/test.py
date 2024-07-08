from config import Config
from mobile_gpt import GPT
from tqdm import tqdm
import torch

cfg = Config.from_name('mobilellm')
gpt = GPT(config=cfg)

print(gpt)

for i in tqdm(range(10000)):
    idx = torch.Tensor(range(1024)).reshape(1, -1).to(torch.long)
    gpt.forward(idx)