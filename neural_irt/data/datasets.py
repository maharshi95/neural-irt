import os
from typing import Any, Optional, Sequence

import datasets as hf_datasets
import torch
from torch.utils.data import Dataset as TorchDataset

from neural_irt.configs.common import DatasetConfig

StringDict = dict[str, Any]
StateDict = dict[str, torch.Tensor]


def load_as_hf_dataset(name_or_path: str) -> hf_datasets.Dataset:
    if name_or_path.endswith(".json") or name_or_path.endswith(".jsonl"):
        return hf_datasets.load_dataset("json", data_files=name_or_path)["train"]
    if os.path.isdir(name_or_path):
        return hf_datasets.load_from_disk(name_or_path)
    else:
        return hf_datasets.load_dataset(name_or_path)


class IrtDataset(TorchDataset):
    def __init__(
        self,
        responses: Sequence[StringDict],
        queries: Sequence[StringDict],
        agents: Sequence[StringDict],
        query_input_format: str = "id",
        agent_input_format: str = "id",
        query_embeddings: Optional[StateDict] = None,
        agent_embeddings: Optional[StateDict] = None,
    ):
        # Check that the query/agent input formats are valid
        if query_input_format not in ["id", "embedding", "text"]:
            raise ValueError(f"Unknown query input format: {query_input_format}")
        if agent_input_format not in ["id", "embedding", "text"]:
            raise ValueError(f"Unknown agent input format: {agent_input_format}")
        self.question_input_format = query_input_format
        self.agent_input_format = agent_input_format

        self.queries = {entry["id"]: entry for entry in queries}
        self.agents = {entry["id"]: entry for entry in agents}
        self.responses = responses

        response_query_ids = {r["query_id"] for r in responses}
        response_agent_ids = {r["agent_id"] for r in responses}

        # Check that all query/agent ids are present in the respective datasets
        if response_query_ids - self.queries.keys():
            raise ValueError("All query ids must be present in the query dataset")
        if response_agent_ids - self.agents.keys():
            raise ValueError("All agent ids must be present in the agent dataset")

        if query_input_format == "embedding":
            self.query_embeddings = query_embeddings or {
                e["id"]: e["embedding"] for e in queries
            }
            # Check that all query ids are present in the query embeddings
            if response_query_ids - self.query_embeddings.keys():
                raise ValueError(
                    "All query ids must be present in the query embeddings"
                )
        if agent_input_format == "embedding":
            self.agent_embeddings = agent_embeddings or {
                e["id"]: e["embedding"] for e in agents
            }
            # Check that all agent ids are present in the agent embeddings
            if response_agent_ids - self.agent_embeddings.keys():
                raise ValueError(
                    "All agent ids must be present in the agent embeddings"
                )

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, index) -> StringDict:
        response = self.responses[index]
        query_id = response["query_id"]
        agent_id = response["agent_id"]

        entry = {"ruling": response["ruling"]}

        if self.question_input_format == "id":
            entry["query_id"] = query_id
        elif self.question_input_format == "embedding":
            entry["query_rep"] = self.query_embeddings[query_id]
        elif self.question_input_format == "text":
            entry["query_text"] = self.queries[query_id]["text"]
        else:
            raise ValueError(
                f"Unknown question input format: {self.question_input_format}"
            )

        if self.agent_input_format == "id":
            entry["agent_id"] = agent_id
        elif self.agent_input_format == "embedding":
            entry["agent_rep"] = self.agents[agent_id]["embedding"]
        elif self.agent_input_format == "text":
            entry["agent_text"] = self.agents[agent_id]["text"]
        else:
            raise ValueError(f"Unknown agent input format: {self.agent_input_format}")

        return entry


def load_embeddings(path: Optional[str]) -> Optional[StateDict]:
    if path is None:
        return None
    return torch.load(path)


def load_irt_dataset(
    dataset_config: DatasetConfig,
    query_input_format: str,
    agent_input_format: str,
    query_embeddings_path: Optional[str] = None,
    agent_embeddings_path: Optional[str] = None,
) -> IrtDataset:
    queries = load_as_hf_dataset(dataset_config.queries).with_format("torch")
    agents = load_as_hf_dataset(dataset_config.agents).with_format("torch")
    responses = load_as_hf_dataset(dataset_config.responses)
    query_embeddings = load_embeddings(query_embeddings_path)
    agent_embeddings = load_embeddings(agent_embeddings_path)
    return IrtDataset(
        responses,
        queries,
        agents,
        query_input_format,
        agent_input_format,
        query_embeddings,
        agent_embeddings,
    )
