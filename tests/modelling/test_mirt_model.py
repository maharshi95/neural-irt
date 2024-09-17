import os
import tempfile
import unittest

import torch

from neural_irt.modeling.mirt import MirtConfig, MirtModel, MirtModelOutput


class TestNeuralMIRTModel(unittest.TestCase):
    def setUp(self):
        self.config = MirtConfig(n_agents=10, n_agent_types=3, n_items=20, n_dim=7)
        self.model = MirtModel(config=self.config)
        self.bs = 8

    def test_model_initialization(self):
        model = MirtModel(config=self.config)
        self.assertIsInstance(model, MirtModel)

    def test_model_components(self):
        self.assertEqual(self.model.agent_embeddings.weight.shape, (10, 7))
        self.assertEqual(self.model.agent_type_embeddings.weight.shape, (3, 7))
        self.assertIn("diff", self.model.item_characteristics)
        self.assertIn("disc", self.model.item_characteristics)
        self.assertIsInstance(self.model.guess_bias, torch.nn.Parameter)

        if self.config.characteristics_bounder:
            self.assertIsInstance(self.model.bounder, torch.nn.Module)

    def test_forward_pass(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_ids = torch.randint(0, 20, (self.bs,))
        agent_type_ids = torch.randint(0, 3, (self.bs,))

        output = self.model(agent_ids, item_ids, agent_type_ids)
        self.assertIsInstance(output, MirtModelOutput)
        self.assertEqual(output.logits.shape, (self.bs,))
        self.assertEqual(output.difficulty.shape, (self.bs, 1))
        self.assertEqual(output.discriminability.shape, (self.bs, 7))
        self.assertEqual(output.skill.shape, (self.bs, 7))

    def test_compute_functions(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_ids = torch.randint(0, 20, (self.bs,))
        agent_type_ids = torch.randint(0, 3, (self.bs,))

        skills = self.model.compute_agent_skills(agent_ids, agent_type_ids)
        self.assertEqual(skills.shape, (self.bs, 7))

        agent_type_skills = self.model.compute_agent_type_skills(agent_type_ids)
        self.assertEqual(agent_type_skills.shape, (self.bs, 7))

        discriminability = self.model.compute_discriminability(item_ids)
        self.assertEqual(discriminability.shape, (self.bs, 7))

        characteristics = self.model.compute_item_characteristics(item_ids)
        self.assertIn("difficulty", characteristics)
        self.assertIn("discriminability", characteristics)
        self.assertEqual(characteristics["difficulty"].shape, (self.bs, 1))
        self.assertEqual(characteristics["discriminability"].shape, (self.bs, 7))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "ckpt")

            # Test save_pretrained
            self.model.save_pretrained(ckpt_dir)
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "model.pt")))

            # Test load_pretrained
            loaded_model = MirtModel.load_pretrained(ckpt_dir)
            self.assertIsInstance(loaded_model, MirtModel)

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
            agent_ids = torch.randint(0, 10, (self.bs,))
            item_ids = torch.randint(0, 20, (self.bs - 1,))
            self.model(agent_ids, item_ids)

        # Test without agent_type_ids when fit_agent_type_embeddings is True
        self.model.config.fit_agent_type_embeddings = True
        with self.assertRaises(ValueError):
            agent_ids = torch.randint(0, 10, (self.bs,))
            item_ids = torch.randint(0, 20, (self.bs,))
            self.model(agent_ids, item_ids)

    def test_compute_logits(self):
        agent_skills = torch.randn(self.bs, 7)
        item_chars = {
            "difficulty": torch.randn(self.bs, 1),
            "discriminability": torch.abs(torch.randn(self.bs, 7)),
        }
        logits = self.model._compute_logits(agent_skills, item_chars)
        self.assertEqual(logits.shape, (self.bs,))


if __name__ == "__main__":
    unittest.main()
