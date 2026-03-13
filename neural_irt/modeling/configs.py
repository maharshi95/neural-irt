from typing import Optional

from pydantic import BaseModel


class BoundingConfig(BaseModel):
    min_value: float = 0.0
    max_value: float = 1.0
    strategy: str = "sigmoid"  # [sigmoid, clamp, none]

    @property
    def value_range(self):
        return self.max_value - self.min_value

    @property
    def to_string(self):
        range_str = (
            f"[{self.min_value},{self.max_value}]"
            if self.min_value != 0.0 and self.max_value != 1.0
            else f"max={self.max_value}"
            if self.max_value != 1.0
            else ""
        )
        return f"bound-by-{self.strategy}-{range_str}"


class IrtModelConfig(BaseModel):
    n_dim: int
    # Number of agents
    # If not provided, will be inferred from the data (agent_indexer)
    n_agents: Optional[int] = None

    # Number of agent types (e.g. humans, cbqa, ret)
    # If not provided, will be inferred from the data (agent_indexer)
    n_agent_types: Optional[int] = None

    # Boolean flags for trainable parameters
    fit_agent_type_embeddings: bool = False

    fit_guess_bias: bool = False
    characteristics_bounder: Optional[BoundingConfig] = None

    @property
    def arch(self):
        raise NotImplementedError("IrtModelConfig.arch must be implemented")


class MirtConfig(IrtModelConfig):
    # Number of items
    n_items: int

    @property
    def arch(self):
        return "mirt"


class CaimiraConfig(IrtModelConfig):
    # Number of dimensions in item embeddings
    n_dim_item_embed: int

    # Number of dimensions for the agent embedding
    rel_mode: str = "linear"  # [linear, mlp]
    dif_mode: str = "linear"  # [linear, mlp]

    # Number of hidden units for the MLPs if mode is mlp
    n_hidden_dif: int = 128
    n_hidden_rel: int = 128

    # Sparsity controls for importance [only used if fit_importance is True]
    # Temperature for importance
    rel_temperature: float = 0.5

    @property
    def arch(self):
        return "caimira"


class HpcirtConfig(IrtModelConfig):
    """Config for Hierarchical Partially Compensatory IRT (HPCIRT) model.

    This model extends MIRT with:
    - Bifactor agent structure: general factor g + domain factors θ
    - Per-item compensation parameter μ interpolating between compensatory and conjunctive
    - Neural parameterization of item characteristics from embeddings
    """

    # Item embedding dimension (from external encoder, e.g. sentence-transformers)
    n_dim_item_embed: int

    # Neural head modes for item parameter computation [linear, mlp]
    disc_mode: str = "linear"
    diff_mode: str = "linear"
    conj_diff_mode: str = "linear"
    comp_mode: str = "linear"

    # Hidden units for MLP heads (shared across all MLP heads)
    n_hidden: int = 128

    # Relevance weight mode for conjunctive component
    # "derived": r_ik = |α_ik| / max_k |α_ik| (Option A, fewer params)
    # "learned": separate neural head (Option B, more flexible)
    relevance_mode: str = "derived"

    # Prior logit for compensation parameter ω_i
    # Negative values bias μ_i = σ(ω_i) toward 0 (compensatory)
    mu_prior_logit: float = -2.0

    # Whether to apply bifactor hierarchy regularization: ||θ - Λg||²
    bifactor_reg: bool = True

    @property
    def arch(self):
        return "hpcirt"
