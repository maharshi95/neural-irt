import os
import tempfile

import pytest

from neural_irt.data.indexers import AgentIndexer, Indexer


def test_indexer_initialization():
    # Test initialization with items
    indexer = Indexer(["a", "b", "c"])
    assert len(indexer) == 3
    assert list(indexer) == ["a", "b", "c"]

    # Test initialization without items
    empty_indexer = Indexer()
    assert len(empty_indexer) == 0


def test_indexer_add_and_get_id():
    indexer = Indexer()

    # Test adding items
    assert indexer.add("a") == 0
    assert indexer.add("b") == 1
    assert indexer.add("c") == 2

    # Test getting ids
    assert indexer.get_id("a") == 0
    assert indexer.get_id("b") == 1
    assert indexer.get_id("c") == 2

    # Test adding duplicate item
    assert indexer.add("a") == 0

    # Test strict mode
    strict_indexer = Indexer(strict=True)
    strict_indexer.add("x")
    with pytest.raises(ValueError):
        strict_indexer.add("x")


def test_indexer_extend():
    indexer = Indexer()
    indexer.extend(["a", "b", "c"])
    assert len(indexer) == 3
    assert indexer.get_ids(["a", "b", "c"]) == [0, 1, 2]


def test_indexer_get_ids():
    indexer = Indexer(["a", "b", "c"])
    assert indexer.get_ids(["a", "b", "a", "c"]) == [0, 1, 0, 2]
    with pytest.raises(KeyError):
        indexer.get_ids(["d"])


def test_indexer_get_items():
    indexer = Indexer(["a", "b", "c"])
    assert indexer.get_items([0, 1, 0, 2]) == ["a", "b", "a", "c"]
    with pytest.raises(IndexError):
        indexer.get_items([3])


def test_indexer_save_and_load():
    indexer = Indexer(["a", "b", "c"])

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "test_indexer.json")
        indexer.save_to_disk(file_path)
        loaded_indexer = Indexer.load_from_disk(file_path)

    assert loaded_indexer.get_ids(["a", "b", "c"]) == [0, 1, 2]
    assert list(loaded_indexer) == ["a", "b", "c"]


def test_agent_indexer():
    agent_indexer = AgentIndexer(
        name_indexer_or_list=["agent1", "agent2", "agent3"],
        type_indexer_or_list=["type1", "type2"],
        agent_type_map={"agent1": "type1", "agent2": "type1", "agent3": "type2"},
    )

    assert agent_indexer.n_agents == 3
    assert agent_indexer.n_agent_types == 2

    agent_ids, agent_type_ids = agent_indexer(["agent1", "agent2", "agent3"])
    assert agent_ids == [0, 1, 2]
    assert agent_type_ids == [0, 0, 1]

    assert agent_indexer.get_agent_type("agent1") == "type1"
    assert agent_indexer.get_agent_type("agent2") == "type1"
    assert agent_indexer.get_agent_type("agent3") == "type2"
    assert agent_indexer.get_agent_type("unknown_agent") == "<unk>"


def test_agent_indexer_save_and_load():
    agent_indexer = AgentIndexer(
        name_indexer_or_list=["agent1", "agent2"],
        type_indexer_or_list=["type1", "type2"],
        agent_type_map={"agent1": "type1", "agent2": "type2"},
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        agent_indexer.save_to_disk(temp_dir)
        loaded_agent_indexer = AgentIndexer.load_from_disk(temp_dir)

    assert loaded_agent_indexer.n_agents == agent_indexer.n_agents
    assert loaded_agent_indexer.n_agent_types == agent_indexer.n_agent_types
    assert loaded_agent_indexer.agent_type_map == agent_indexer.agent_type_map

    loaded_agent_ids, loaded_agent_type_ids = loaded_agent_indexer(["agent1", "agent2"])
    assert loaded_agent_ids == [0, 1]
    assert loaded_agent_type_ids == [0, 1]
