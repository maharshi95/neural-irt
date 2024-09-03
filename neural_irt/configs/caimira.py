from typing import Optional

from neural_irt.modeling.configs import CaimiraConfig

from . import common


class TrainerConfig(common.TrainerConfig):
    # Coefficients for regularization
    c_reg_skill: float = 1e-6
    c_reg_difficulty: float = 1e-6
    c_reg_relevance: float = 1e-6


class RunConfig(common.RunConfig):
    model: CaimiraConfig
    trainer: TrainerConfig
