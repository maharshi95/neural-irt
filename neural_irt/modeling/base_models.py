import dataclasses
import os
from typing import Optional

import torch
from loguru import logger
from torch import Tensor, nn

from neural_irt.modeling.configs import IrtModelConfig
from neural_irt.modeling.layers import Bounder, create_zero_init_embedding
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


class PretrainedModel(nn.Module):
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

    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        device = resolve_device(device)

        # Load the model config, model weights
        config_path = os.path.join(path, "config.json")
        config = config_utils.load_config(config_path, cls=cls.config_class)

        ckpt_path = os.path.join(path, "model.pt")
        model = cls(config=config)
        model.load_ckpt(ckpt_path, map_location=device)
        return model


class BaseIrtModel(PretrainedModel):
    """Base class for all IRT models."""

    config_class: type[IrtModelConfig]
    output_class: type[IrtModelOutput]

    def __init__(self, config: IrtModelConfig):
        super().__init__()

        self.config = config

        logger.info(f"Model Config: {config}")
        self._build_model()

    def _build_agent_layers(self):
        raise NotImplementedError(
            "BaseIrtModel._build_agent_layers must be implemented"
        )

    def _build_item_layers(self):
        raise NotImplementedError("BaseIrtModel._build_item_layers must be implemented")

    def _build_model(self):
        self._build_agent_layers()
        self._build_item_layers()

        # Setup Guess Bias
        self.guess_bias = nn.Parameter(
            torch.zeros(1), requires_grad=self.config.fit_guess_bias
        )

        if self.config.characteristics_bounder:
            self.bounder = Bounder(self.config.characteristics_bounder)

    def compute_item_characteristics(self, item_inputs: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError(
            "BaseIrtModel.compute_item_characteristics must be implemented"
        )

    def compute_agent_skills(
        self, agent_inputs: Tensor, agent_type_inputs: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError(
            "BaseIrtModel.compute_agent_skills must be implemented"
        )

    def _compute_logits(
        self, agent_skills: Tensor, item_characteristics: dict[str, Tensor]
    ) -> Tensor:
        raise NotImplementedError("BaseIrtModel._compute_logits must be implemented")

    def forward(
        self,
        agent_inputs: Tensor,
        item_inputs: Tensor,
        agent_type_inputs: Optional[Tensor] = None,
    ) -> IrtModelOutput:
        if self.config.fit_agent_type_embeddings and agent_type_inputs is None:
            raise ValueError(
                "Agent type inputs must be provided if config.fit_agent_type_embeddings is True"
            )

        item_chars = self.compute_item_characteristics(item_inputs)
        agent_skills = self.compute_agent_skills(agent_inputs, agent_type_inputs)

        logits = self._compute_logits(agent_skills, item_chars)
        return self.output_class(
            logits=logits,
            skill=agent_skills,
            **item_chars,
        )


class AgentIndexedIrtModel(BaseIrtModel):
    """Base class for all IRT models."""

    def _build_agent_layers(self):
        self.agent_embeddings = nn.Embedding(self.config.n_agents, self.config.n_dim)
        self.agent_embeddings.weight.data.normal_(0, 0.001)

        self.agent_type_embeddings = create_zero_init_embedding(
            n=self.config.n_agent_types,
            dim=self.config.n_dim,
            dtype=torch.float32,
            requires_grad=self.config.fit_agent_type_embeddings,
        )

    def compute_agent_type_skills(self, agent_type_inputs: Tensor):
        """Computes agent type skill weights.

        Args:
            agent_type_inputs (Tensor): Agent type IDs of shape (batch_size,)

        Returns:
            Tensor: Agent type skill vector of shape (batch_size, n_dim)
        """

        skills = self.agent_type_embeddings(agent_type_inputs)

        if self.config.characteristics_bounder:
            skills = self.bounder(skills)
        return skills

    def compute_agent_skills(
        self, agent_inputs: Tensor, agent_type_inputs: Optional[Tensor] = None
    ):
        """Computes agent skill weights.

        Args:
            agent_inputs (Tensor): Agent IDs of shape (batch_size,)
            agent_type_inputs (Optional[Tensor]): Agent type IDs of shape (batch_size,)
        Returns:
            Tensor: Agent skill vector of shape (batch_size, n_dim)
        """
        skills = self.agent_embeddings(agent_inputs)

        if self.config.fit_agent_type_embeddings:
            if agent_type_inputs is None:
                raise ValueError(
                    "Agent type inputs must be provided if "
                    "config.fit_agent_type_embeddings is True"
                )
            skills += self.agent_type_embeddings(agent_type_inputs)
        # elif agent_type_inputs is not None:
        #     # Warn if agent_type_ids are provided but not used
        #     logger.warning(
        #         "Agent type inputs provided but fit_agent_type_embeddings is False. "
        #         "Ignoring agent_type_inputs."
        #     )

        if self.config.characteristics_bounder:
            skills = self.bounder(skills)
        return skills
