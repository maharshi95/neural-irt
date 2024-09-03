import os
from typing import Any, Sequence

import datasets
from torch.utils.data import Dataset

StringDict = dict[str, Any]


def load_dataset_hf(dataset_name_or_path: str):
    if dataset_name_or_path.endswith(".json") or dataset_name_or_path.endswith(
        ".jsonl"
    ):
        return datasets.load_dataset("json", data_files=dataset_name_or_path)["train"]
    if os.path.isdir(dataset_name_or_path):
        return datasets.load_from_disk(dataset_name_or_path)
    else:
        return datasets.load_dataset(dataset_name_or_path)


class IrtDataset(Dataset):
    def __init__(
        self,
        responses: Sequence[StringDict],
        queries: Sequence[StringDict],
        agents: Sequence[StringDict],
        query_input_format: str = "id",
        agent_input_format: str = "id",
    ):
        self.queries = {entry["id"]: entry for entry in queries}
        self.agents = {entry["id"]: entry for entry in agents}
        self.responses = responses
        self.question_input_format = query_input_format
        self.agent_input_format = agent_input_format

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
            entry["query_rep"] = self.queries[query_id]["embedding"]
        elif self.question_input_format == "text":
            entry["query_text"] = self.queries[query_id]["text"]
        else:
            raise ValueError(
                f"Unknown question input format: {self.question_input_format}"
            )

        if self.agent_input_format == "id":
            entry["agent_id"] = agent_id
        elif self.agent_input_format == "embedding":
            entry["agent_rep"] = self.queries[agent_id]["embedding"]
        elif self.agent_input_format == "text":
            entry["agent_text"] = self.queries[agent_id]["text"]
        else:
            raise ValueError(f"Unknown agent input format: {self.agent_input_format}")

        return entry
