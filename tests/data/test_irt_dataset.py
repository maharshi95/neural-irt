"""This module contains tests for the datasets module."""

import tempfile
import unittest

import pytest
import torch

from neural_irt.configs.common import DatasetConfig
from neural_irt.data import datasets

# Mock data for testing
mock_responses = [
    {"query_id": "q1", "agent_id": "a1", "ruling": 1},
    {"query_id": "q2", "agent_id": "a2", "ruling": 0},
]

mock_queries = [
    {"id": "q1", "text": "Query 1", "embedding": [0.1, 0.2, 0.3]},
    {"id": "q2", "text": "Query 2", "embedding": [0.4, 0.5, 0.6]},
]

mock_agents = [
    {"id": "a1", "text": "Agent 1", "embedding": torch.tensor([0.7, 0.8, 0.9])},
    {"id": "a2", "text": "Agent 2", "embedding": torch.tensor([1.0, 1.1, 1.2])},
]


@pytest.fixture
def irt_dataset():
    return datasets.IrtDataset(mock_responses, mock_queries, mock_agents)


class TestIrtDataset(unittest.TestCase):
    def test_irt_dataset_initialization(self):
        dataset = datasets.IrtDataset(mock_responses, mock_queries, mock_agents)
        self.assertEqual(len(dataset), len(mock_responses))
        self.assertEqual(dataset.queries, {q["id"]: q for q in mock_queries})
        self.assertEqual(dataset.agents, {a["id"]: a for a in mock_agents})

    def test_irt_dataset_getitem_id_format(self):
        dataset = datasets.IrtDataset(mock_responses, mock_queries, mock_agents)
        entry = dataset[0]
        self.assertEqual(entry["ruling"], 1)
        self.assertEqual(entry["query_id"], "q1")
        self.assertEqual(entry["agent_id"], "a1")

    def test_irt_dataset_getitem_embedding_format(self):
        dataset = datasets.IrtDataset(
            mock_responses, mock_queries, mock_agents, "embedding", "embedding"
        )
        entry = dataset[0]
        self.assertEqual(entry["ruling"], 1)
        self.assertTrue(torch.equal(entry["agent_rep"], torch.tensor([0.7, 0.8, 0.9])))
        self.assertTrue(entry["query_rep"], [0.1, 0.2, 0.3])

    def test_irt_dataset_getitem_text_format(self):
        dataset = datasets.IrtDataset(
            mock_responses, mock_queries, mock_agents, "text", "text"
        )
        entry = dataset[0]
        self.assertEqual(entry["ruling"], 1)
        self.assertEqual(entry["query_text"], "Query 1")
        self.assertEqual(entry["agent_text"], "Agent 1")

    def test_irt_dataset_invalid_format(self):
        with self.assertRaises(ValueError):
            datasets.IrtDataset(
                mock_responses, mock_queries, mock_agents, "invalid", "id"
            )
        with self.assertRaises(ValueError):
            datasets.IrtDataset(
                mock_responses, mock_queries, mock_agents, "id", "invalid"
            )


@pytest.fixture
def mock_dataset_config():
    return DatasetConfig(
        queries="mock_queries.json",
        agents="mock_agents.json",
        responses="mock_responses.json",
    )


def test_load_as_hf_dataset(mocker):
    mock_load_dataset = mocker.patch("datasets.load_dataset")
    mock_load_from_disk = mocker.patch("datasets.load_from_disk")

    # Test JSON file
    datasets.load_as_hf_dataset("test.json")
    mock_load_dataset.assert_called_with("json", data_files="test.json")

    # Test dataset name
    datasets.load_as_hf_dataset("test_dataset")
    mock_load_dataset.assert_called_with("test_dataset")

    # Test directory
    temp_dir = tempfile.TemporaryDirectory()
    datasets.load_as_hf_dataset(temp_dir.name)
    mock_load_from_disk.assert_called_with(temp_dir.name)
    temp_dir.cleanup()


def test_load_irt_dataset(mocker, mock_dataset_config):
    mock_load_as_hf_dataset = mocker.patch(
        "neural_irt.data.datasets.load_as_hf_dataset"
    )
    mock_load_as_hf_dataset.return_value.with_format.return_value = mock_queries

    dataset = datasets.load_irt_dataset(mock_dataset_config, "id", "id")

    assert isinstance(dataset, datasets.IrtDataset)
    assert mock_load_as_hf_dataset.call_count == 3


if __name__ == "__main__":
    unittest.main()
