import dataclasses
import os
from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn

from neural_irt.modeling.configs import IrtModelConfig
from neural_irt.utils import config_utils


def resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    elif device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


@dataclasses.dataclass
class IrtModelOutput:
    logits: Tensor
    difficulty: Tensor
    skill: Tensor


class NeuralIrtModel(nn.Module):
    """Base class for all IRT models."""

    config_class: type[IrtModelConfig]

    def __init__(self, config: IrtModelConfig):
        super().__init__()

        self.config = config

        logger.info(f"Model Config: {config}")
        self._build_model()

    def _build_model(self):
        # Implement this method to build the model architecture
        raise NotImplementedError("NeuralIrtModel._build_model must be implemented")

    def forward(self, *args, **kwargs) -> IrtModelOutput:
        # Implement this method to define the forward pass
        raise NotImplementedError("NeuralIrtModel.forward must be implemented")

    def save_ckpt(self, path: str):
        torch.save(self.state_dict(), path)

    def load_ckpt(self, path: str, map_location: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location))

    def save_pretrained(self, path: str):
        # Save the model config, model weights

        os.makedirs(path, exist_ok=True)

        # Save model config
        config_path = os.path.join(path, "config.json")
        config_utils.save_config(self.config, config_path)

        # Save model weights
        weights_path = os.path.join(path, "model.pt")
        self.save_ckpt(weights_path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        device = resolve_device(device)

        # Load the model config, model weights
        config_path = os.path.join(path, "config.json")
        config = config_utils.load_config(config_path, cls=cls.config_class)

        ckpt_path = os.path.join(path, "model.pt")
        model = cls(config=config)
        model.load_ckpt(ckpt_path, map_location=device)
        logger.info(f"Model loaded from {path}")
        return model
