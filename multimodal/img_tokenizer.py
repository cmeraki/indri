# Copyright (c) Meta Platforms, Inc. and affiliates
#
# This source code is licensed under the Chameleon License found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import PIL
import torch
import yaml
from PIL import Image

from vqgan import VQModel
from torchvision.transforms import v2


class ImageTokenizer:
    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        device: str | torch.device | None = None,
    ):
        with open(cfg_path) as f:
            config = yaml.safe_load(f)

        params = config["model"]["params"]
        if "lossconfig" in params:
            del params["lossconfig"]
        params["ckpt_path"] = ckpt_path

        self._vq_model = VQModel(**params)
        self._vq_model.eval()

        if device is None:
            devices = {p.device for p in self._vq_model.parameters()}
            assert len(devices) == 1
            device = devices.pop()
        else:
            self._vq_model.to(device)

        self._vq_model = torch.compile(self._vq_model)
        print(self._vq_model)
        self._device = device

        self.transforms = v2.Compose([
            v2.Resize(size=512),
            v2.CenterCrop(size=(512, 512)),
            v2.ToDtype(torch.float16)
        ])

        dtypes = {p.dtype for p in self._vq_model.parameters()}
        assert len(dtypes) == 1
        self._dtype = dtypes.pop()

    def _whiten_transparency(self, img: PIL.Image) -> PIL.Image:
        # Check if it's already in RGB format.
        if img.mode == "RGB":
            return img

        vals_rgba = np.array(img.convert("RGBA"))

        # If there is no transparency layer, simple convert and return.
        if not (vals_rgba[:, :, 3] < 255).any():
            return img.convert("RGB")

        # There is a transparency layer, blend it with a white background.

        # Calculate the alpha proportion for blending.
        alpha = vals_rgba[:, :, 3] / 255.0
        # Blend with white background.
        vals_rgb = (1 - alpha[:, :, np.newaxis]) * 255 + alpha[
            :, :, np.newaxis
        ] * vals_rgba[:, :, :3]
        return PIL.Image.fromarray(vals_rgb.astype("uint8"), "RGB")

    def img_tokens_from_pil(self, image: torch.Tensor) -> list[int]:
        image = (self.transforms(image) / 255.) * 2 - 1.0
        image = image.unsqueeze(0)
        _, _, [_, _, img_toks] = self._vq_model.encode(image)
        return img_toks

    def _pil_from_chw_tensor(self, chw_tensor: torch.Tensor) -> PIL.Image:
        # Ensure detachment and move tensor to CPU.
        detached_chw_tensor = chw_tensor.detach().cpu()

        # Normalize tensor to [0, 1] range from [-1, 1] range.
        normalized_chw_tensor = (
            torch.clamp(detached_chw_tensor, -1.0, 1.0) + 1.0
        ) / 2.0

        # Permute CHW tensor to HWC format and convert to NumPy array.
        hwc_array = normalized_chw_tensor.permute(1, 2, 0).numpy()

        # Convert to an 8-bit unsigned integer format.
        image_array_uint8 = (hwc_array * 255).astype(np.uint8)

        # Convert NumPy array to PIL Image.
        pil_image = Image.fromarray(image_array_uint8)

        # Convert image to RGB if it is not already.
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        return pil_image

    def pil_from_img_toks(self, img_tensor: torch.Tensor) -> PIL.Image:
        emb_dim = self._vq_model.quantize.embedding.weight.shape[-1]
        codebook_entry = self._vq_model.quantize.get_codebook_entry(
            img_tensor, (1, 32, 32, emb_dim)
        )
        pixels = self._vq_model.decode(codebook_entry)
        return self._pil_from_chw_tensor(pixels[0])
