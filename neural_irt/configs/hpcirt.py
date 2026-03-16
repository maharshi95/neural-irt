from neural_irt.modeling.configs import HpcirtConfig

from . import common


class TrainerConfig(common.TrainerConfig):
    # Regularization coefficients
    c_reg_skill: float = 1e-6  # L1 on domain factors θ
    c_reg_difficulty: float = 1e-6  # L1 on compensatory difficulty
    c_reg_g: float = 1e-6  # L2 on general factor g
    c_reg_mu: float = 1e-4  # Penalty encouraging μ toward 0 or 1 (bimodal)
    c_reg_disc: float = 1e-6  # L1 sparsity on discrimination vectors
    c_reg_bifactor: float = 1e-4  # Bifactor hierarchy: ||θ - Λg||²


class RunConfig(common.RunConfig):
    model: HpcirtConfig
    trainer: TrainerConfig
