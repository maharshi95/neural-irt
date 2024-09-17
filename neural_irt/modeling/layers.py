from typing import Any

import torch
from torch import Tensor, nn

from .configs import BoundingConfig


class Bounder(nn.Module):
    def __init__(self, config: BoundingConfig):
        super(Bounder, self).__init__()
        self.config = config

    def forward(self, x: Tensor):
        if self.config.strategy == "sigmoid":
            return self.config.min_value + torch.sigmoid(x) * (self.config.value_range)
        elif self.config.strategy == "clamp":
            return torch.clamp(x, self.config.min_value, self.config.max_value)
        elif self.config.strategy == "none":
            return x
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")


def create_zero_init_embedding(
    n: int, dim: int, dtype: Any = torch.float32, requires_grad: bool = True
):
    embedding = nn.Embedding(n, dim, _weight=torch.zeros((n, dim), dtype=dtype))
    embedding.weight.requires_grad = requires_grad
    return embedding
