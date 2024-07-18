
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type, Union

import torch
import yaml
from typing_extensions import Self

def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class Config:
    name: str = ""
    hf_config: dict = field(default_factory=dict)
    scale_embeddings: bool = False
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_head: int = 32
    head_size: Optional[int] = None
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    lm_head_bias: bool = False
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    norm_class_name: Literal["LayerNorm", "RMSNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    mlp_class_name: Literal["GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE"] = "GptNeoxMLP"
    gelu_approximate: str = "none"
    intermediate_size: Optional[int] = None
    rope_condense_ratio: int = 1
    rope_base: int = 10000
    n_expert: int = 0
    n_expert_per_token: int = 0
    n_layer_repeat: int = 1

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_config.get("name", self.name)

        if self.head_size is None:
            assert self.n_embd % self.n_head == 0
            self.head_size = self.n_embd // self.n_head

        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, self.padding_multiple)
        else:
            # vocab size shouldn't be larger than padded vocab size
            self.vocab_size = min(self.vocab_size, self.padded_vocab_size)

        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self.mlp_class_name == "LLaMAMLP":
                raise ValueError(f"The config {self.name!r}, needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

        self.rope_n_elem = int(self.rotary_percentage * self.head_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Optional[Self]:
        if name not in name_to_config:
            # search through all `config['hf_config']['name']`
            try:
                conf_dict = next(
                    config for config in configs
                    if name == config["hf_config"]["name"] or
                    config["hf_config"]["org"] + "/" + config["hf_config"]["name"] == name
                )
            except StopIteration:
                raise ValueError(f"{name!r} is not a supported config name")
        else:
            conf_dict = name_to_config[name]

        conf_dict = conf_dict.copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs: Any) -> Self:
        with open(path, encoding="utf-8") as fp:
            file_kwargs = yaml.safe_load(fp)
            if file_kwargs is None:
                raise ValueError(f"{path} is empty which is likely unexpected.")
        file_kwargs.update(kwargs)
        return cls(**file_kwargs)

    @classmethod
    def from_checkpoint(cls, path: Path, **kwargs: Any) -> Self:
        """Automatically load `model_config.yaml` and if it doesn't exist - a matching config from `litgpt/config.py`."""
        if (config_path := path / "model_config.yaml").is_file():
            return cls.from_file(config_path, **kwargs)
        if (model_name := path.name) in name_to_config:
            return cls.from_name(model_name, **kwargs)
        raise FileNotFoundError(f"For {str(path)!r} neither 'model_config.yaml' nor matching config exists.")


configs = [
    dict(
        name="mobilellm",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=30,
        n_head=16,
        n_embd=576,
        n_layer_repeat=2,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama use FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="LLaMAMLP",
        intermediate_size=1536,
        n_query_groups=4,
    ),
]

name_to_config = {config["name"]: config for config in configs}