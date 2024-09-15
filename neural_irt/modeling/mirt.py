# %%

import dataclasses
from typing import Optional, Sequence

import torch
from loguru import logger
from torch import Tensor, nn

from neural_irt.modeling.base_models import AgentIndexedIrtModel, IrtModelOutput
from neural_irt.modeling.layers import create_zero_init_embedding

from .configs import MirtConfig


@dataclasses.dataclass
class MirtModelOutput(IrtModelOutput):
    # Below fields are inherited from IrtModelOutput
    # logits: Tensor
    # difficulty: Tensor
    # skill: Tensor
    discriminability: Tensor


class MirtModel(AgentIndexedIrtModel):
    config_class = MirtConfig
    output_class = MirtModelOutput

    def __init__(self, config: MirtConfig):
        super().__init__(config)

    def _build_item_layers(self):
        self.item_characteristics = nn.ModuleDict(
            {
                "diff": create_zero_init_embedding(self.config.n_items, 1),
                "disc": nn.Embedding(self.config.n_items, self.config.n_dim),
            }
        )

    def compute_discriminability(self, item_ids: Tensor) -> Tensor:
        disc_weights = self.item_characteristics["disc"](item_ids)
        return torch.abs(disc_weights)

    def compute_item_characteristics(self, item_ids: Tensor) -> dict[str, Tensor]:
        characteristics = {
            "difficulty": self.item_characteristics["diff"](item_ids),
            "discriminability": self.compute_discriminability(item_ids),
        }
        return characteristics

    def _compute_logits(self, agent_skills, item_chars):
        disc = item_chars["discriminability"]
        diff = item_chars["difficulty"]

        logits = torch.einsum("bn,bn->b", disc, agent_skills) + diff[:, 0]

        if self.config.fit_guess_bias:
            logits += self.guess_bias
        return logits

    def forward(
        self,
        agent_ids: Tensor,
        item_ids: Tensor,
        agent_type_ids: Optional[Tensor] = None,
    ) -> MirtModelOutput:
        """Compute logits for each agent-item pair."""
        # agent_ids: (batch_size,)
        # item_embeddings: (batch_size, n_dim_item_embed)
        # agent_type_ids: Optional[(batch_size,)]

        return super().forward(agent_ids, item_ids, agent_type_ids)
