import os
from typing import Any, Optional, Sequence

import datasets as hf_datasets
import torch
from torch.utils.data import Dataset as TorchDataset

from neural_irt.configs.common import DatasetConfig

StringDict = dict[str, Any]
StateDict = dict[str, torch.Tensor]


def process_dataset_name(name: str) -> tuple[str, Optional[str], Optional[str]]:
    # split the name with :
    # example {dataset_name}:{config_name}:{split_name}
    # it could also be {dataset_name}:{split_name} or {dataset_name}::{split_name}
    parts = name.split(":")
    if len(parts) == 1:
        return parts[0], None, None
    elif len(parts) == 2:
        return parts[0], None, parts[1]
    elif len(parts) == 3:
        if parts[1] == "":
            return parts[0], None, parts[2]
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(
            f"Invalid dataset name format: {name}. "
            "Allowed formats are {dataset_name} or {dataset_name}:{config_name} "
            "or {dataset_name}:{config_name}:{split_name}"
        )


def load_hf_dataset(
    name_or_path: str,
    config_name: Optional[str] = None,
    split_name: Optional[str] = None,
) -> hf_datasets.Dataset:
    if os.path.isdir(name_or_path):
        if config_name is not None:
            raise ValueError("Cannot specify config name when loading from disk.")
        return hf_datasets.load_from_disk(name_or_path, split=split_name)
    else:
        return hf_datasets.load_dataset(
            name_or_path, name=config_name, split=split_name
        )


def load_as_hf_dataset(name_or_path: str) -> hf_datasets.Dataset:
    if name_or_path.endswith(".json") or name_or_path.endswith(".jsonl"):
        return hf_datasets.load_dataset("json", data_files=name_or_path)["train"]
    name_or_path, config_name, split_name = process_dataset_name(name_or_path)
    return load_hf_dataset(name_or_path, config_name, split_name)


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
        if query_input_format == "embedding" and not (
            query_embeddings or "embedding" in queries[0]
        ):
            raise ValueError(
                "Query embeddings must be provided when using embedding format. "
                "Either provide the embeddings or ensure that the queries have an "
                "'embedding' field"
            )
        if agent_input_format == "embedding" and not (
            agent_embeddings or "embedding" in agents[0]
        ):
            raise ValueError(
                "Agent embeddings must be provided when using embedding format. "
                "Either provide the embeddings or ensure that the agents have an "
                "'embedding' field"
            )

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
            entry["agent_rep"] = self.agents[agent_id]["embedding"]
        elif self.agent_input_format == "text":
            entry["agent_text"] = self.agents[agent_id]["text"]
            entry["agent_text"] = self.agents[agent_id]["text"]
        else:
            raise ValueError(f"Unknown agent input format: {self.agent_input_format}")

        return entry


def load_embeddings(path: Optional[str]) -> Optional[StateDict]:
    if path is None:
        return None
    embeds = torch.load(path)
    # Check if the embeds are a state dict. Check if the values are tensors, and if
    # not, convert them to tensors
    if not isinstance(embeds, dict):
        raise ValueError(f"Embeddings must be a state dict. Got a {type(embeds)}")

    if isinstance(next(iter(embeds.values())), torch.Tensor):
        return embeds
    return {k: torch.tensor(v) for k, v in embeds.items()}


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
