"""
This module provides indexing functionality for items and agents.

Classes:
    Indexer: A generic serializable class for mapping items to unique IDs
    AgentIndexer: A class (subclass of Indexer) for assigning unique integer IDs to agent names and types.

Functions:
    add: Adds an item to the indexer and returns its ID.
    extend: Adds multiple items to the indexer.
"""

# %%
import json
import os
from typing import Any, Callable, Generic, Iterable, Optional, Sequence, TypeVar, Union

import numpy as np
import torch

T = TypeVar("T")


class Indexer(Generic[T]):
    def __init__(
        self,
        items: Optional[Iterable[T]] = None,
        strict: bool = False,
        key: Optional[Callable[[T], Any]] = None,
    ):
        self._item_to_id: dict[Any, int] = {}
        self._id_to_item: list[T] = []
        self.strict = strict
        self.key_fn = key
        if items is not None:
            self.extend(items)

    def __len__(self) -> int:
        return len(self._item_to_id)

    def __iter__(self):
        return iter(self._id_to_item)

    def __contains__(self, item: T) -> bool:
        key = self.key_fn(item) if self.key_fn else item
        return key in self._item_to_id

    def __getitem__(self, index: Union[int, slice]) -> Union[T, list[T]]:
        return self._id_to_item[index]

    def add(self, item: T) -> int:
        """Add an item to the indexer and return its ID."""
        key = self.key_fn(item) if self.key_fn else item
        if key in self._item_to_id:
            if self.strict:
                raise ValueError(
                    f"Duplicate item '{item}' found in strict mode. "
                    "Use strict=False if you want the Indexer to ignore duplicates."
                )
            return self._item_to_id[key]
        id_ = len(self._item_to_id)
        self._item_to_id[key] = id_
        self._id_to_item.append(item)
        return id_

    def extend(self, items: Iterable[T]) -> None:
        """Add multiple items to the indexer."""
        for item in items:
            self.add(item)

    def get_id(self, item: T) -> int:
        key = self.key_fn(item) if self.key_fn else item
        return self._item_to_id[key]

    def get_ids(self, items: Iterable[T]) -> list[int]:
        return [self.get_id(item) for item in items]

    def get_items(self, ids: Iterable[int]) -> list[T]:
        return [self._id_to_item[i] for i in ids]

    def __call__(self, items: Sequence[T], return_tensors: Optional[str] = None):
        ids = self.get_ids(items)
        if return_tensors == "pt":
            return torch.tensor(ids)
        elif return_tensors == "np":
            return np.array(ids)
        return ids

    @property
    def data_dict(self) -> dict:
        return {"id_to_item": self._id_to_item}

    def state_dict(self) -> dict:
        return self.data_dict

    def save_to_disk(self, filepath: str) -> None:
        # Save the indexer to a file
        with open(filepath, "w") as f:
            json.dump(self.data_dict, f)

    @classmethod
    def load_from_data_dict(cls, data_dict: dict) -> "Indexer[T]":
        indexer = cls()
        indexer._id_to_item = data_dict["id_to_item"]
        indexer._item_to_id = {e: i for i, e in enumerate(indexer._id_to_item)}
        return indexer

    @classmethod
    def load_from_disk(cls, filepath: str) -> "Indexer[T]":
        # Load the indexer from a file
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.load_from_data_dict(data)


class AgentIndexer:
    def __init__(
        self,
        name_indexer_or_list: Indexer[str] | list[str] | None = None,
        type_indexer_or_list: Indexer[str] | list[str] | None = None,
        agent_type_map: dict[str, str] | None = None,
        strict: bool = False,
        use_unk_type: bool = False,
    ):
        self.name_indexer: Indexer[str] = (
            name_indexer_or_list
            if isinstance(name_indexer_or_list, Indexer)
            else Indexer(name_indexer_or_list, strict=strict)
        )

        self.type_indexer: Indexer[str] = (
            type_indexer_or_list
            if isinstance(type_indexer_or_list, Indexer)
            else Indexer(type_indexer_or_list, strict=strict)
        )

        if use_unk_type and "<unk>" not in self.type_indexer:
            self.type_indexer.add("<unk>")

        self.agent_type_map: dict[str, str] = agent_type_map or {}

    def get_agent_ids(self, agent_names: Sequence[str]):
        return self.name_indexer.get_ids(agent_names)

    def get_agent_type_ids(self, agent_types: Sequence[str]):
        return self.type_indexer.get_ids(agent_types)

    def get_agent_type(self, agent_names: str | Sequence | str):
        if isinstance(agent_names, str):
            return self.agent_type_map.get(agent_names, "<unk>")
        return [self.agent_type_map.get(name, "<unk>") for name in agent_names]

    def map_to_ids(self, agent_names: Sequence[str], agent_types: Sequence[str] = None):
        agent_types = agent_types or self.get_agent_type(agent_names)
        return self.get_agent_ids(agent_names), self.get_agent_type_ids(agent_types)

    def __call__(
        self, agent_names: Sequence[str], return_tensors: Optional[str] = None
    ):
        agent_ids, agent_types = self.map_to_ids(agent_names)
        if return_tensors == "pt":
            return torch.tensor(agent_ids), torch.tensor(agent_types)
        elif return_tensors == "np":
            return np.array(agent_ids), np.array(agent_types)
        return agent_ids, agent_types

    @property
    def n_agents(self):
        return len(self.name_indexer)

    @property
    def n_agent_types(self):
        return len(self.type_indexer)

    @property
    def data_dict(self):
        return {
            "names": self.name_indexer.data_dict,
            "types": self.type_indexer.data_dict,
            "type_map": self.agent_type_map,
        }

    def state_dict(self):
        return self.data_dict

    def save_to_disk(self, dirpath: str):
        # Save the indexer to a file
        os.makedirs(dirpath, exist_ok=True)
        for k, v in self.data_dict.items():
            filepath = os.path.join(dirpath, f"agent_indexer.{k}.json")
            with open(filepath, "w") as fp:
                json.dump(v, fp)

    @classmethod
    def load_from_data_dict(cls, data_dict: dict):
        name_indexer = Indexer.load_from_data_dict(data_dict["names"])
        type_indexer = Indexer.load_from_data_dict(data_dict["types"])
        return cls(name_indexer, type_indexer, data_dict["type_map"])

    @classmethod
    def load_from_disk(cls, dirpath: str):
        # Load the indexer from a file
        data_dict = {}
        for k in ["names", "types", "type_map"]:
            filepath = os.path.join(dirpath, f"agent_indexer.{k}.json")
            with open(filepath, "r") as f:
                data_dict[k] = json.load(f)
        return cls.load_from_data_dict(data_dict)


# Example usage
if __name__ == "__main__":
    responses: list[str] = ["a", "b", "c"]
    indexer = Indexer(responses)

    # Convert items to IDs
    items = ["a", "b", "a", "c"]
    ids = indexer.get_ids(items)
    print("Items to IDs:", ids)

    # Convert IDs back to items
    items_converted_back = indexer.get_items(ids)
    print("IDs to Items:", items_converted_back)

    # Save the indexer to a file
    indexer.save_to_disk("indexer.json")

    # Load the indexer from a file
    loaded_indexer = Indexer.load_from_disk("indexer.json")

    # Convert items to IDs using the loaded indexer
    loaded_ids = loaded_indexer.get_ids(items)
    print("Loaded Indexer - Items to IDs:", loaded_ids)

    # Add new items to the indexer
    new_items = ["d", "e"]
    indexer.extend(new_items)
    new_ids = indexer.get_ids(["d", "e", "a"])
    print("New Items to IDs:", new_ids)

# %%
