from typing import Any, Dict, List

import torch

from neural_irt.data.indexers import AgentIndexer, Indexer


class NeuralMirtCollator:
    def __init__(
        self,
        agent_indexer: AgentIndexer,
        query_indexer: Indexer[str],
        is_training: bool = True,
    ):
        self.agent_indexer = agent_indexer
        self.query_indexer = query_indexer
        self.is_training = is_training

    def __call__(self, entries: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        agent_names = [entry["agent_name"] for entry in entries]
        query_ids = [entry["query_id"] for entry in entries]
        rulings = [entry["ruling"] for entry in entries]

        agent_ids, agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        item_ids = self.query_indexer(query_ids, return_tensors="pt")

        batch = {
            "agent_ids": agent_ids,
            "agent_type_ids": agent_type_ids,
            "item_ids": item_ids,
            "labels": torch.tensor(rulings, dtype=torch.float32),
        }

        if not self.is_training:
            for key in entries[0].keys():
                if key not in ["agent_name", "agent_type", "query_id", "ruling"]:
                    batch[key] = [entry[key] for entry in entries]

        return batch


class CaimiraCollator:
    def __init__(self, agent_indexer: AgentIndexer, is_training: bool = True):
        self.agent_indexer = agent_indexer
        self.is_training = is_training

    def __call__(self, entries: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        agent_names = [entry["agent_id"] for entry in entries]
        agent_ids, agent_type_ids = self.agent_indexer(agent_names, return_tensors="pt")
        item_embeddings = [entry["query_rep"] for entry in entries]
        item_embeddings = (
            torch.stack(item_embeddings) if item_embeddings else torch.empty(0)
        )

        batch = {
            "agent_ids": agent_ids,
            "agent_type_ids": agent_type_ids,
            "item_embeddings": item_embeddings,
            "labels": torch.tensor(
                [entry["ruling"] for entry in entries], dtype=torch.float32
            ),
        }

        if not self.is_training:
            for key in entries[0].keys():
                if key not in ["agent_id", "query_rep", "ruling"]:
                    batch[key] = [entry[key] for entry in entries]

        return batch
