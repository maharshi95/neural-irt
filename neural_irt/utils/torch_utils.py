from typing import Optional

import torch


def resolve_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cpu")
    elif device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)
