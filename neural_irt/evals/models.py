# %%
import torch
from torch import Tensor, nn

from neural_irt.data.indexers import AgentIndexer
from neural_irt.modeling.caimira import CaimiraConfig, CaimiraModel
from neural_irt.modeling.hpcirt import HpcirtModel


class CaimiraInferenceModel(nn.Module):
    def __init__(self, model: CaimiraModel, agent_indexer: AgentIndexer):
        super().__init__()
        self.model = model
        self.agent_indexer = agent_indexer
        self.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def forward(self, agent_names, item_embeddings, agent_types=None):
        agent_ids = self.agent_indexer.get_agent_ids(agent_names)
        agent_type_ids = (
            self.agent_indexer.get_agent_type_ids(agent_types)
            if agent_types is not None
            else None
        )
        return self.model.forward(agent_ids, item_embeddings, agent_type_ids)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.agent_indexer.save_to_disk(path)

    def compute_agent_skills(self, agent_names):
        agent_ids, agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        with torch.no_grad():
            return self.model.compute_agent_skills(agent_ids, agent_type_ids)

    def compute_agent_type_skills(self, agent_names):
        agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        with torch.no_grad():
            return self.model.compute_agent_type_skills(agent_type_ids)

    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        model = CaimiraModel.load_pretrained(path, device=device)
        agent_indexer = AgentIndexer.load_from_disk(path)
        return cls(model=model, agent_indexer=agent_indexer)


class HpcirtInferenceModel(nn.Module):
    """Inference wrapper for HPCIRT model with agent name resolution."""

    def __init__(self, model: HpcirtModel, agent_indexer: AgentIndexer):
        super().__init__()
        self.model = model
        self.agent_indexer = agent_indexer
        self.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def forward(self, agent_names, item_embeddings, agent_types=None):
        agent_ids = self.agent_indexer.get_agent_ids(agent_names)
        agent_type_ids = (
            self.agent_indexer.get_agent_type_ids(agent_types)
            if agent_types is not None
            else None
        )
        return self.model.forward(agent_ids, item_embeddings, agent_type_ids)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)
        self.agent_indexer.save_to_disk(path)

    def compute_agent_skills(self, agent_names) -> dict[str, Tensor]:
        """Compute bifactor agent skills: general factor g and domain factors θ.

        Returns:
            Dict with 'g' (n_agents, 1) and 'theta' (n_agents, K)
        """
        agent_ids, agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        with torch.no_grad():
            return self.model.compute_agent_skills(agent_ids, agent_type_ids)

    def compute_mu_distribution(self, item_embeddings: Tensor) -> Tensor:
        """Compute compensation parameter μ for a batch of items.

        Args:
            item_embeddings: Item embeddings, shape (N, n_dim_item_embed)

        Returns:
            μ values in [0, 1], shape (N, 1). Values near 0 = compensatory, near 1 = conjunctive.
        """
        with torch.no_grad():
            chars = self.model.compute_item_characteristics(item_embeddings)
            return chars["mu"]

    @classmethod
    def load_pretrained(cls, path: str, device: str = "auto"):
        model = HpcirtModel.load_pretrained(path, device=device)
        agent_indexer = AgentIndexer.load_from_disk(path)
        return cls(model=model, agent_indexer=agent_indexer)


# %%
checkpoint_path = "checkpoints/irt/sample_run/epoch_10"


# %%
model = CaimiraInferenceModel.load_pretrained(checkpoint_path, device="cpu")

# %%
skills = model.compute_agent_skills(["a1", "a2", "a3", "a4", "a5", "a6", "a7"])
type_skills = model.compute_agent_type_skills(
    ["a1", "a2", "a3", "a4", "a5", "a6", "a7"]
)
# %%
model.model.agent_embeddings(2)
# %%
