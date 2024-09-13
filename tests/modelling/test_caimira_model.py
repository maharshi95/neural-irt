"""This module contains tests for the models module."""

import os
import shutil
import tempfile
import unittest

import torch

from neural_irt.modeling.caimira import CaimiraConfig, CaimiraModel, CaimiraModelOutput


class TestCaimiraModel(unittest.TestCase):
    def setUp(self):
        self.config = CaimiraConfig(
            n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32
        )
        self.model = CaimiraModel(config=self.config)
        self.bs = 8

    def test_model_initialization(self):
        # Test valid initialization
        model = CaimiraModel(config=self.config)
        self.assertIsInstance(model, CaimiraModel)

        # Test invalid initialization
        with self.assertRaises(ValueError):
            invalid_config = CaimiraConfig(
                n_agents=10,
                n_agent_types=3,
                n_dim=7,
                n_dim_item_embed=32,
                dif_mode="invalid_mode",
            )
            CaimiraModel(config=invalid_config)

    def test_model_components(self):
        self.assertEqual(self.model.agent_embeddings.weight.shape, (10, 7))
        self.assertEqual(self.model.agent_type_embeddings.weight.shape, (3, 7))
        self.assertIsInstance(self.model.layer_dif, torch.nn.Module)
        self.assertIsInstance(self.model.layer_rel, torch.nn.Module)
        self.assertIsInstance(self.model.guess_bias, torch.nn.Parameter)

        if self.config.characteristics_bounder:
            self.assertIsInstance(self.model.bounder, torch.nn.Module)

    def test_forward_pass(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)
        agent_type_ids = torch.randint(0, 3, (self.bs,))

        output = self.model(agent_ids, item_embeddings, agent_type_ids)
        self.assertIsInstance(output, CaimiraModelOutput)
        self.assertEqual(output.logits.shape, (self.bs,))
        self.assertEqual(output.difficulty.shape, (self.bs, 7))
        self.assertEqual(output.relevance.shape, (self.bs, 7))
        self.assertEqual(output.skill.shape, (self.bs, 7))

    def test_compute_functions(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        agent_type_ids = torch.randint(0, 3, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        skills = self.model.compute_agent_skills(agent_ids, agent_type_ids)
        self.assertEqual(skills.shape, (self.bs, 7))

        agent_type_skills = self.model.compute_agent_type_skills(agent_type_ids)
        self.assertEqual(agent_type_skills.shape, (self.bs, 7))

        difficulty = self.model.compute_item_difficulty(item_embeddings)
        self.assertEqual(difficulty.shape, (self.bs, 7))

        relevance = self.model.compute_item_relevance(item_embeddings)
        self.assertEqual(relevance.shape, (self.bs, 7))
        self.assertTrue(torch.allclose(relevance.sum(dim=1), torch.ones(self.bs)))

        characteristics = self.model.compute_item_characteristics(item_embeddings)
        self.assertIn("difficulty", characteristics)
        self.assertIn("relevance", characteristics)
        self.assertEqual(characteristics["difficulty"].shape, (self.bs, 7))
        self.assertEqual(characteristics["relevance"].shape, (self.bs, 7))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "ckpt")

            # Test save_pretrained
            self.model.save_pretrained(ckpt_dir)
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "model.pt")))

            # Test load_pretrained
            loaded_model = CaimiraModel.load_pretrained(ckpt_dir)
            self.assertIsInstance(loaded_model, CaimiraModel)

            # Check if loaded model has the same parameters as the original
            for param1, param2 in zip(
                self.model.parameters(), loaded_model.parameters()
            ):
                self.assertTrue(torch.equal(param1, param2))

    def test_edge_cases(self):
        # Test with empty inputs
        with self.assertRaises(RuntimeError):
            self.model(torch.tensor([]), torch.tensor([]))

        # Test with invalid input shapes
        with self.assertRaises(RuntimeError):
            self.model(torch.randint(0, 10, (self.bs,)), torch.randn(self.bs - 1, 32))

        # Test without agent_type_ids when fit_agent_type_embeddings is True
        self.model.config.fit_agent_type_embeddings = True
        with self.assertRaises(ValueError):
            self.model(torch.randint(0, 10, (self.bs,)), torch.randn(self.bs, 32))

    def test_compute_logits(self):
        agent_skills = torch.randn(self.bs, 7)
        item_chars = {
            "difficulty": torch.randn(self.bs, 7),
            "relevance": torch.softmax(torch.randn(self.bs, 7), dim=1),
        }
        logits = self.model._compute_logits(agent_skills, item_chars)
        self.assertEqual(logits.shape, (self.bs,))


if __name__ == "__main__":
    unittest.main()
