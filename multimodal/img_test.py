import torch
import glob

torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

from torchvision.io import read_image, ImageReadMode
from img_tokenizer import ImageTokenizer
from tqdm import tqdm

cfg_path = '../models/chameleon_tokenizer/vqgan.yaml'
ckpt_path = '../models/chameleon_tokenizer/vqgan.ckpt'

tokenizer = ImageTokenizer(cfg_path=cfg_path, ckpt_path=ckpt_path, device='cuda:0')


with torch.inference_mode():
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for i in tqdm(glob.glob('../data/images/data/images/download/00000/*.jpg')):
            image = read_image(i, mode=ImageReadMode.RGB)
            image = image.to('cuda:0')
            tokens = tokenizer.img_tokens_from_pil(image)
#
# image = tokenizer.pil_from_img_toks(torch.clone(tokens))
# image.save('test.png')

# print(list(tokens.cpu().numpy()))
# image = tokenizer.pil_from_img_toks(tokens)
# image.save('test.png')