import pytest
import torch

from neural_irt.data.collators import CaimiraCollator, NeuralMirtCollator
from neural_irt.data.indexers import AgentIndexer, Indexer


@pytest.fixture
def agent_indexer():
    return AgentIndexer(
        ["agent1", "agent2", "agent3"],
        ["type1", "type2"],
        agent_type_map={"agent1": "type1", "agent2": "type2", "agent3": "type1"},
    )


@pytest.fixture
def query_indexer():
    return Indexer(["query1", "query2", "query3"])


@pytest.fixture
def neural_mirt_entries():
    return [
        {
            "agent_name": "agent1",
            "query_id": "query1",
            "ruling": 1,
            "extra_key": "value1",
        },
        {
            "agent_name": "agent2",
            "query_id": "query2",
            "ruling": 0,
            "extra_key": "value2",
        },
        {
            "agent_name": "agent3",
            "query_id": "query3",
            "ruling": 1,
            "extra_key": "value3",
        },
    ]


@pytest.fixture
def caimira_entries():
    return [
        {
            "agent_id": "agent1",
            "query_rep": torch.randn(10),
            "ruling": 1.0,
            "extra_key": "value1",
        },
        {
            "agent_id": "agent2",
            "query_rep": torch.randn(10),
            "ruling": 0.0,
            "extra_key": "value2",
        },
        {
            "agent_id": "agent3",
            "query_rep": torch.randn(10),
            "ruling": 1.0,
            "extra_key": "value3",
        },
    ]


def test_neural_mirt_collator(agent_indexer, query_indexer, neural_mirt_entries):
    collator = NeuralMirtCollator(agent_indexer, query_indexer)
    batch = collator(neural_mirt_entries)

    required_keys = ["agent_ids", "agent_type_ids", "item_ids", "labels"]

    for key in required_keys:
        assert key in batch
        assert batch[key].shape[0] == len(neural_mirt_entries)
        if key == "labels":
            assert batch[key].dtype == torch.float32
        else:
            assert batch[key].dtype == torch.int64


def test_caimira_collator(agent_indexer, caimira_entries):
    collator = CaimiraCollator(agent_indexer)
    batch = collator(caimira_entries)

    required_keys = ["agent_ids", "agent_type_ids", "item_embeddings", "labels"]
    for key in required_keys:
        assert key in batch
        assert batch[key].shape[0] == len(caimira_entries)
        if key in ["labels", "item_embeddings"]:
            assert batch[key].dtype == torch.float32
        else:
            assert batch[key].dtype == torch.int64


def test_neural_mirt_collator_empty_input(agent_indexer, query_indexer):
    collator = NeuralMirtCollator(agent_indexer, query_indexer)
    batch = collator([])

    assert all(tensor.numel() == 0 for tensor in batch.values())


def test_caimira_collator_empty_input(agent_indexer):
    collator = CaimiraCollator(agent_indexer)
    batch = collator([])

    assert all(tensor.numel() == 0 for tensor in batch.values())


def test_neural_mirt_collator_invalid_input(agent_indexer, query_indexer):
    collator = NeuralMirtCollator(agent_indexer, query_indexer)
    with pytest.raises(KeyError):
        collator([{"invalid_key": "value"}])


def test_caimira_collator_invalid_input(agent_indexer):
    collator = CaimiraCollator(agent_indexer)
    with pytest.raises(KeyError):
        collator([{"invalid_key": "value"}])


def test_neural_mirt_collator_non_training_mode(
    agent_indexer, query_indexer, neural_mirt_entries
):
    collator = NeuralMirtCollator(agent_indexer, query_indexer, is_training=False)
    batch = collator(neural_mirt_entries)

    for key in neural_mirt_entries[0].keys():
        if key not in ["agent_name", "query_id", "ruling"]:
            assert key in batch
            assert isinstance(batch[key], list)
            assert len(batch[key]) == len(neural_mirt_entries)


def test_caimira_collator_non_training_mode(agent_indexer, caimira_entries):
    collator = CaimiraCollator(agent_indexer, is_training=False)
    batch = collator(caimira_entries)

    for key in caimira_entries[0].keys():
        if key not in ["agent_id", "query_rep", "ruling"]:
            assert key in batch
            assert isinstance(batch[key], list)
            assert len(batch[key]) == len(caimira_entries)
