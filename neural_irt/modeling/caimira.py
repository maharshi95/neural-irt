# %%

import dataclasses
import os
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from neural_irt.utils import config_utils

from .configs import CaimiraConfig
from .layers import Bounder


@dataclasses.dataclass
class CaimiraModelOutput:
    logits: Tensor
    difficulty: Tensor
    relevance: Tensor
    skill: Tensor


def create_zero_init_embedding(
    n: int, dim: int, dtype: Any = torch.float32, requires_grad: bool = True
):
    embedding = nn.Embedding(n, dim, _weight=torch.zeros((n, dim), dtype=dtype))
    embedding.weight.requires_grad = requires_grad
    return embedding


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device)


class CaimiraModel(nn.Module):
    def __init__(self, config: CaimiraConfig):
        super().__init__()

        self.config = config

        logger.info(f"Model Config: {config}")
        self._build_model()

    def _build_model(self):
        self.agent_embeddings = nn.Embedding(self.config.n_agents, self.config.n_dim)
        self.agent_embeddings.weight.data.normal_(0, 0.001)

        self.agent_type_embeddings = create_zero_init_embedding(
            n=self.config.n_agent_types,
            dim=self.config.n_dim,
            dtype=torch.float32,
            requires_grad=self.config.fit_agent_type_embeddings,
        )

        # Setup Item difficulty
        if self.config.dif_mode == "linear":
            self.layer_dif = nn.Linear(self.config.n_dim_item_embed, self.config.n_dim)
        elif self.config.dif_mode == "mlp":
            self.layer_dif = nn.Sequential(
                nn.Dropout(p=0.05),
                nn.Linear(self.config.n_dim_item_embed, self.config.n_hidden_dif),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.config.n_hidden_dif, self.config.n_dim),
            )
        else:
            raise ValueError(f"Unknown difficulty mode {self.config.dif_mode}")

        # Setup Item relevance
        if self.config.rel_mode == "linear":
            self.layer_rel = nn.Linear(self.config.n_dim_item_embed, self.config.n_dim)
        elif self.config.rel_mode == "mlp":
            self.layer_rel = nn.Sequential(
                nn.Dropout(p=0.05),
                nn.Linear(self.config.n_dim_item_embed, self.config.n_hidden_rel),
                nn.LeakyReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.config.n_hidden_rel, self.config.n_dim),
            )
        else:
            raise ValueError(f"Unknown relevance mode {self.config.rel_mode}")

        # Setup Guess Bias
        self.guess_bias = nn.Parameter(
            torch.zeros(1), requires_grad=self.config.fit_guess_bias
        )

        if self.config.characteristics_bounder:
            self.bounder = Bounder(self.config.characteristics_bounder)

    def compute_agent_skills(
        self, agent_ids: Tensor, agent_type_ids: Optional[Tensor] = None
    ):
        """
        Calculates the skill weights for the agent based on the given subjects and subject types.

        Args:
            agent_ids (Tensor): The agent_ids for which to calculate the skill weights.
            agent_type_ids (Optional[Tensor]): The agent types for which to calculate the skill weights.

        Returns:
            Tensor: The skill vector for the agent.
        """

        skills = self.agent_embeddings(agent_ids)

        if self.config.fit_agent_type_embeddings:
            if agent_type_ids is None:
                raise ValueError(
                    "Agent type ids must be provided if config.fit_agent_type_embeddings is True"
                )
            skills += self.agent_type_embeddings(agent_type_ids)

        if self.config.characteristics_bounder:
            skills = self.bounder(skills)
        return skills

    def compute_agent_type_skills(self, agent_type_ids: Tensor):
        """
        Calculates the skill weights for the agent type based on the given agent types.

        Args:
            agent_type_ids (Tensor): The agent types for which to calculate the skill weights.

        Returns:
            Tensor: The skill vector for the agent type.
        """

        skills = self.agent_type_embeddings(agent_type_ids)

        if self.config.characteristics_bounder:
            skills = self.bounder(skills)
        return skills

    def compute_item_difficulty(self, item_embeddings: Tensor) -> Tensor:
        """Returns the relative difficulty of the items."""
        dif_raw = self.layer_dif(item_embeddings)
        dif_norm = dif_raw - dif_raw.mean(dim=0)

        if self.config.characteristics_bounder:
            dif_norm = self.bounder(dif_norm)
        return dif_norm

    def compute_item_relevance(self, item_embeddings: Tensor) -> Tensor:
        rel_raw = self.layer_rel(item_embeddings)
        # rel_norm = F.layer_norm(rel_raw)

        # Better than sigmoid since it helps normalize across the number of dimensions
        # Also better regularizes the importance weights to have generate sparse probs.
        rel_norm = F.softmax(rel_raw / self.config.rel_temperature, dim=-1)

        return rel_norm

    def compute_item_characteristics(
        self, item_embeddings: Sequence[Tensor]
    ) -> dict[str, Tensor]:
        item_relevance = self.compute_item_relevance(item_embeddings)
        item_difficulty = self.compute_item_difficulty(item_embeddings)
        characteristics = {
            "difficulty": item_difficulty,
            "relevance": item_relevance,
        }

        return characteristics

    def _compute_logits(self, agent_skills, item_chars):
        latent_scores = agent_skills - item_chars["difficulty"]
        # logits = torch.sum((skills - difficulty) * relevance, dim=1, keepdim=True)
        logits = torch.einsum("bn,bn->b", latent_scores, item_chars["relevance"])

        if self.config.fit_guess_bias:
            logits += self.guess_bias
        return logits

    def forward(
        self,
        agent_ids: Tensor,
        item_embeddings: Tensor,
        agent_type_ids: Optional[Tensor] = None,
    ) -> CaimiraModelOutput:
        """Compute logits for each agent-item pair."""
        # agent_ids: (batch_size,)
        # item_embeddings: (batch_size, n_dim_item_embed)
        # agent_type_ids: Optional[(batch_size,)]

        if self.config.fit_agent_type_embeddings and agent_type_ids is None:
            raise ValueError(
                "Agent type ids must be provided if config.fit_agent_type_embeddings is True"
            )

        item_chars = self.compute_item_characteristics(item_embeddings)
        agent_skills = self.compute_agent_skills(agent_ids, agent_type_ids)

        logits = self._compute_logits(agent_skills, item_chars)
        return CaimiraModelOutput(
            logits=logits,
            **item_chars,
            skill=agent_skills,
        )

    def save_pretrained(self, path: str):
        # Save the model config, model weights

        os.makedirs(path, exist_ok=True)

        # Save model config
        config_path = os.path.join(path, "config.json")
        config_utils.save_config(self.config, config_path)

        # Save model weights
        weights_path = os.path.join(path, "model.pt")
        torch.save(self.state_dict(), weights_path)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        device = resolve_device(device)

        # Load the model config, model weights
        config_path = os.path.join(path, "config.json")
        config = config_utils.load_config(config_path, cls=CaimiraConfig)

        ckpt_path = os.path.join(path, "model.pt")
        model = cls(config=config)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    config = CaimiraConfig(n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32)
    model = CaimiraModel(config=config)

    batch_size = 64
    agent_ids = torch.randint(0, 10, (batch_size,))
    item_embeddings = torch.randn(batch_size, 32)
    agent_type_ids = torch.randint(0, 3, (batch_size,))

    output = model(agent_ids, item_embeddings, agent_type_ids)
    print("Output logits shape:", output.logits.shape)
    print("Output difficulty shape:", output.difficulty.shape)
    print("Output relevance shape:", output.relevance.shape)
    print("Output skill shape:", output.skill.shape)
