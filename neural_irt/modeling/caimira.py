import dataclasses
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_irt.modeling.base_models import AgentIndexedIrtModel, IrtModelOutput

from .configs import CaimiraConfig


@dataclasses.dataclass
class CaimiraModelOutput(IrtModelOutput):
    # Below fields are inherited from IrtModelOutput
    # logits: Tensor
    # difficulty: Tensor
    # skill: Tensor
    relevance: Tensor


class CaimiraModel(AgentIndexedIrtModel):
    config_class = CaimiraConfig
    output_class = CaimiraModelOutput

    def __init__(self, config: CaimiraConfig):
        super().__init__(config)

    def _build_item_layers(self):
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

    def compute_item_difficulty(self, item_embeddings: Tensor) -> Tensor:
        """Computes the relative difficulty of the items.

        Args:
            item_embeddings (Tensor): Item embeddings of shape (batch_size, n_dim_item_embed)

        Returns:
            Tensor: Item difficulty vector of shape (batch_size, n_dim)
        """
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

        return super().forward(agent_ids, item_embeddings, agent_type_ids)


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
