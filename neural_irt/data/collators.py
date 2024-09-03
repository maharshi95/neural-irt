from typing import Any, Dict, List

import torch

from neural_irt.data.indexers import AgentIndexer, Indexer


class NeuralMirtCollator:
    def __init__(
        self,
        agent_indexer: AgentIndexer,
        item_indexer: Indexer[str],
        is_training: bool = True,
    ):
        self.agent_indexer = agent_indexer
        self.item_indexer = item_indexer
        self.is_training = is_training

    def __call__(self, entries: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        agent_names = [entry["agent_name"] for entry in entries]
        agent_types = [entry["agent_type"] for entry in entries]
        query_ids = [entry["query_id"] for entry in entries]
        rulings = [entry["ruling"] for entry in entries]

        agent_ids = self.agent_indexer.get_agent_ids(agent_names)
        agent_type_ids = self.agent_indexer.get_agent_type_ids(agent_types)
        item_ids = self.item_indexer.get_ids(query_ids)

        return {
            "agent_ids": torch.tensor(agent_ids, dtype=torch.int),
            "agent_type_ids": torch.tensor(agent_type_ids, dtype=torch.int),
            "item_ids": torch.tensor(item_ids, dtype=torch.int),
            "labels": torch.tensor(rulings, dtype=torch.int),
        }


class CaimiraCollator:
    def __init__(self, agent_indexer: AgentIndexer, is_training: bool = True):
        self.agent_indexer = agent_indexer
        self.is_training = is_training

    def __call__(self, entries: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        agent_names = [entry["agent_id"] for entry in entries]
        agent_ids, agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        item_embeddings = [entry["query_rep"] for entry in entries]

        return {
            "agent_ids": agent_ids,
            "agent_type_ids": agent_type_ids,
            "item_embeddings": torch.stack(item_embeddings),
            "labels": torch.tensor(
                [entry["ruling"] for entry in entries], dtype=torch.float32
            ),
        }
