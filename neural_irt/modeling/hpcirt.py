"""Hierarchical Partially Compensatory IRT (HPCIRT) model.

This model extends standard compensatory MIRT with:
- Bifactor agent structure: general factor g + domain-specific factors θ
- Per-item compensation parameter μ_i interpolating between compensatory and conjunctive
- Neural parameterization of item characteristics from pre-computed embeddings

Reference: HPCIRT proposal (Draft v0.1)
"""

import dataclasses
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_irt.modeling.base_models import BaseIrtModel, IrtModelOutput, PretrainedModel
from neural_irt.modeling.layers import Bounder, create_zero_init_embedding

from .configs import HpcirtConfig


@dataclasses.dataclass
class HpcirtModelOutput(IrtModelOutput):
    # Inherited from IrtModelOutput:
    # logits: Tensor
    # difficulty: Tensor  (compensatory difficulty d_i)
    # skill: Tensor       (domain factors θ, shape (B, K))

    # HPCIRT-specific fields
    g: Tensor  # General factor, shape (B, 1)
    disc_g: Tensor  # General factor discrimination, shape (B, 1)
    disc: Tensor  # Domain discrimination α_i, shape (B, K)
    diff_conj: Tensor  # Conjunctive per-dimension difficulty δ_i, shape (B, K)
    diff_conj_g: Tensor  # Conjunctive general difficulty δ_i0, shape (B, 1)
    mu: Tensor  # Compensation parameter μ_i ∈ [0,1], shape (B, 1)


def _make_head(input_dim: int, output_dim: int, mode: str, n_hidden: int) -> nn.Module:
    """Create a linear or MLP head for item parameter computation."""
    if mode == "linear":
        return nn.Linear(input_dim, output_dim)
    elif mode == "mlp":
        return nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(input_dim, n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(n_hidden, output_dim),
        )
    else:
        raise ValueError(f"Unknown head mode: {mode!r}. Expected 'linear' or 'mlp'.")


class HpcirtModel(BaseIrtModel):
    """Hierarchical Partially Compensatory IRT model.

    Agent side (bifactor):
        - g_m ∈ ℝ: general ability factor per agent
        - θ_m ∈ ℝ^K: domain-specific ability vector per agent
        - Optional regularization: ||θ_m - Λ·g_m||² encourages θ to be explained by g

    Item side (neural parameterization from embeddings):
        - a_i^(g): discrimination on general factor (scalar, positive)
        - α_i: discrimination on domain factors (vector, positive)
        - d_i: compensatory difficulty (scalar)
        - δ_i: conjunctive per-dimension difficulty (vector)
        - δ_i0: conjunctive general difficulty (scalar)
        - μ_i: compensation parameter ∈ [0,1] (0=compensatory, 1=conjunctive)

    Response function:
        H = P_comp^(1-μ) · P_conj^μ  (computed in log-space)
    """

    config_class = HpcirtConfig
    output_class = HpcirtModelOutput

    def __init__(self, config: HpcirtConfig):
        # Store config before calling super().__init__ which calls _build_model
        super().__init__(config)

    def _build_agent_layers(self):
        config = self.config

        # General factor: g_m ∈ ℝ (stored as (n_agents, 1))
        self.g_embeddings = nn.Embedding(config.n_agents, 1)
        self.g_embeddings.weight.data.normal_(0, 0.001)

        # Domain factors: θ_m ∈ ℝ^K
        self.theta_embeddings = nn.Embedding(config.n_agents, config.n_dim)
        self.theta_embeddings.weight.data.normal_(0, 0.001)

        # Bifactor loading vector: Λ ∈ ℝ^K (maps g → expected θ)
        self.lambda_loading = nn.Parameter(torch.zeros(config.n_dim))

        # Agent type embeddings (optional, additive on θ)
        self.agent_type_embeddings = create_zero_init_embedding(
            n=config.n_agent_types,
            dim=config.n_dim,
            dtype=torch.float32,
            requires_grad=config.fit_agent_type_embeddings,
        )

    def _build_item_layers(self):
        config = self.config
        d_in = config.n_dim_item_embed
        K = config.n_dim
        h = config.n_hidden

        # Discrimination on general factor: a_i^(g) ∈ ℝ (positive via abs)
        self.layer_disc_g = _make_head(d_in, 1, config.disc_mode, h)

        # Discrimination on domain factors: α_i ∈ ℝ^K (positive via abs)
        self.layer_disc = _make_head(d_in, K, config.disc_mode, h)

        # Compensatory difficulty: d_i ∈ ℝ
        self.layer_diff_comp = _make_head(d_in, 1, config.diff_mode, h)

        # Conjunctive per-dimension difficulty: δ_i ∈ ℝ^K
        self.layer_diff_conj = _make_head(d_in, K, config.conj_diff_mode, h)

        # Conjunctive general difficulty: δ_i0 ∈ ℝ
        self.layer_diff_conj_g = _make_head(d_in, 1, config.conj_diff_mode, h)

        # Compensation parameter: ω_i ∈ ℝ → μ_i = σ(ω_i) ∈ [0,1]
        self.layer_comp = _make_head(d_in, 1, config.comp_mode, h)

        # Initialize compensation head bias toward mu_prior_logit
        # so that μ starts near σ(mu_prior_logit) ≈ 0.12 (compensatory prior)
        self._init_comp_bias(config.mu_prior_logit)

    def _init_comp_bias(self, prior_logit: float):
        """Initialize the compensation head's bias toward the prior logit value."""
        # Find the last Linear layer in layer_comp
        if isinstance(self.layer_comp, nn.Linear):
            self.layer_comp.bias.data.fill_(prior_logit)
        elif isinstance(self.layer_comp, nn.Sequential):
            # Find last Linear in the Sequential
            for module in reversed(list(self.layer_comp.modules())):
                if isinstance(module, nn.Linear):
                    module.bias.data.fill_(prior_logit)
                    break

    def compute_agent_skills(
        self, agent_inputs: Tensor, agent_type_inputs: Optional[Tensor] = None
    ) -> dict[str, Tensor]:
        """Compute bifactor agent skills: general factor g and domain factors θ.

        Args:
            agent_inputs: Agent IDs, shape (B,)
            agent_type_inputs: Optional agent type IDs, shape (B,)

        Returns:
            Dict with 'g' (B, 1) and 'theta' (B, K)
        """
        g = self.g_embeddings(agent_inputs)  # (B, 1)
        theta = self.theta_embeddings(agent_inputs)  # (B, K)

        if self.config.fit_agent_type_embeddings:
            if agent_type_inputs is None:
                raise ValueError(
                    "Agent type inputs must be provided if "
                    "config.fit_agent_type_embeddings is True"
                )
            theta = theta + self.agent_type_embeddings(agent_type_inputs)

        if self.config.characteristics_bounder:
            theta = self.bounder(theta)

        return {"g": g, "theta": theta}

    def compute_item_characteristics(
        self, item_inputs: Tensor
    ) -> dict[str, Tensor]:
        """Compute item parameters from item embeddings.

        Args:
            item_inputs: Item embeddings, shape (B, n_dim_item_embed)

        Returns:
            Dict with disc_g, disc, difficulty (compensatory), diff_conj, diff_conj_g, mu
        """
        disc_g = torch.abs(self.layer_disc_g(item_inputs))  # (B, 1)
        disc = torch.abs(self.layer_disc(item_inputs))  # (B, K)
        difficulty = self.layer_diff_comp(item_inputs)  # (B, 1) — compensatory
        diff_conj = self.layer_diff_conj(item_inputs)  # (B, K)
        diff_conj_g = self.layer_diff_conj_g(item_inputs)  # (B, 1)
        mu = torch.sigmoid(self.layer_comp(item_inputs))  # (B, 1)

        return {
            "disc_g": disc_g,
            "disc": disc,
            "difficulty": difficulty,
            "diff_conj": diff_conj,
            "diff_conj_g": diff_conj_g,
            "mu": mu,
        }

    def _compute_logits(
        self, agent_skills: dict[str, Tensor], item_chars: dict[str, Tensor]
    ) -> Tensor:
        """Compute HPCIRT response logits via hybrid compensatory-conjunctive function.

        The hybrid response function H interpolates between:
        - Compensatory: P_comp = σ(a_g·g + α·θ - d)
        - Conjunctive: P_conj = σ(a_g·g - δ_0) · ∏_k σ(α_k·θ_k - δ_k)^r_k

        H = P_comp^(1-μ) · P_conj^μ, computed in log-space for stability.

        Returns:
            Logits (log-odds) suitable for binary_cross_entropy_with_logits, shape (B,)
        """
        g = agent_skills["g"]  # (B, 1)
        theta = agent_skills["theta"]  # (B, K)

        a_g = item_chars["disc_g"]  # (B, 1)
        alpha = item_chars["disc"]  # (B, K)
        d = item_chars["difficulty"]  # (B, 1)
        delta = item_chars["diff_conj"]  # (B, K)
        delta_0 = item_chars["diff_conj_g"]  # (B, 1)
        mu = item_chars["mu"]  # (B, 1)

        # --- Compensatory component ---
        # P_comp = σ(a_g·g + α·θ - d)
        logit_comp = (
            a_g * g
            + torch.einsum("bk,bk->b", alpha, theta).unsqueeze(-1)
            - d
        )  # (B, 1)
        log_p_comp = F.logsigmoid(logit_comp)  # (B, 1)

        # --- Conjunctive component ---
        # P_conj = σ(a_g·g - δ_0) · ∏_k σ(α_k·θ_k - δ_k)^r_k
        log_p_conj_g = F.logsigmoid(a_g * g - delta_0)  # (B, 1)

        # Per-dimension conjunctive terms
        logit_conj_k = alpha * theta - delta  # (B, K)
        log_p_conj_k = F.logsigmoid(logit_conj_k)  # (B, K)

        # Relevance weights: r_ik = |α_ik| / max_k |α_ik| (Option A: derived)
        alpha_abs = alpha.detach()  # detach to avoid double gradient path through relevance
        r = alpha_abs / alpha_abs.max(dim=-1, keepdim=True).values.clamp(min=1e-8)  # (B, K)

        # Sum weighted log-probabilities across dimensions
        log_p_conj = log_p_conj_g + (r * log_p_conj_k).sum(dim=-1, keepdim=True)  # (B, 1)

        # --- Hybrid combination in log-space ---
        # log H = (1-μ)·log P_comp + μ·log P_conj
        log_h = (1.0 - mu) * log_p_comp + mu * log_p_conj  # (B, 1)

        # Convert log-probability to logit (log-odds) for BCE loss
        # logit = log(H / (1-H)) = log_h - log(1 - exp(log_h))
        # Clamp log_h to avoid log(0) when H ≈ 1
        log_h_clamped = log_h.clamp(max=-1e-7)
        logits = log_h_clamped - torch.log1p(-torch.exp(log_h_clamped))  # (B, 1)

        logits = logits.squeeze(-1)  # (B,)

        if self.config.fit_guess_bias:
            logits = logits + self.guess_bias

        return logits

    def compute_bifactor_reg_loss(self, agent_inputs: Tensor) -> Tensor:
        """Compute bifactor hierarchy regularization: ||θ - Λ·g||².

        Encourages domain factors θ to be explained by the general factor g
        through the loading vector Λ.

        Args:
            agent_inputs: Agent IDs, shape (B,)

        Returns:
            Scalar regularization loss.
        """
        g = self.g_embeddings(agent_inputs)  # (B, 1)
        theta = self.theta_embeddings(agent_inputs)  # (B, K)
        expected_theta = g * self.lambda_loading.unsqueeze(0)  # (B, K)
        return ((theta - expected_theta) ** 2).mean()

    def forward(
        self,
        agent_ids: Tensor,
        item_embeddings: Tensor,
        agent_type_ids: Optional[Tensor] = None,
    ) -> HpcirtModelOutput:
        """Compute HPCIRT response logits for agent-item pairs.

        Args:
            agent_ids: Agent IDs, shape (B,)
            item_embeddings: Pre-computed item embeddings, shape (B, n_dim_item_embed)
            agent_type_ids: Optional agent type IDs, shape (B,)

        Returns:
            HpcirtModelOutput with logits and all intermediate parameters.
        """
        if self.config.fit_agent_type_embeddings and agent_type_ids is None:
            raise ValueError(
                "Agent type inputs must be provided if config.fit_agent_type_embeddings is True"
            )

        agent_skills = self.compute_agent_skills(agent_ids, agent_type_ids)
        item_chars = self.compute_item_characteristics(item_embeddings)
        logits = self._compute_logits(agent_skills, item_chars)

        return HpcirtModelOutput(
            logits=logits,
            difficulty=item_chars["difficulty"],
            skill=agent_skills["theta"],
            g=agent_skills["g"],
            disc_g=item_chars["disc_g"],
            disc=item_chars["disc"],
            diff_conj=item_chars["diff_conj"],
            diff_conj_g=item_chars["diff_conj_g"],
            mu=item_chars["mu"],
        )
