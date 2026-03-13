"""Tests for the HPCIRT (Hierarchical Partially Compensatory IRT) model."""

import os
import tempfile
import unittest

import torch

from neural_irt.modeling.configs import HpcirtConfig
from neural_irt.modeling.hpcirt import HpcirtModel, HpcirtModelOutput


class TestHpcirtModel(unittest.TestCase):
    def setUp(self):
        self.config = HpcirtConfig(
            n_agents=10,
            n_agent_types=3,
            n_dim=5,
            n_dim_item_embed=32,
        )
        self.model = HpcirtModel(config=self.config)
        self.bs = 8

    def test_model_initialization(self):
        model = HpcirtModel(config=self.config)
        self.assertIsInstance(model, HpcirtModel)

        # Test invalid disc_mode
        with self.assertRaises(ValueError):
            invalid_config = HpcirtConfig(
                n_agents=10,
                n_agent_types=3,
                n_dim=5,
                n_dim_item_embed=32,
                disc_mode="invalid_mode",
            )
            HpcirtModel(config=invalid_config)

    def test_model_components(self):
        # Agent layers
        self.assertEqual(self.model.g_embeddings.weight.shape, (10, 1))
        self.assertEqual(self.model.theta_embeddings.weight.shape, (10, 5))
        self.assertEqual(self.model.lambda_loading.shape, (5,))
        self.assertEqual(self.model.agent_type_embeddings.weight.shape, (3, 5))

        # Item layers
        self.assertIsInstance(self.model.layer_disc_g, torch.nn.Module)
        self.assertIsInstance(self.model.layer_disc, torch.nn.Module)
        self.assertIsInstance(self.model.layer_diff_comp, torch.nn.Module)
        self.assertIsInstance(self.model.layer_diff_conj, torch.nn.Module)
        self.assertIsInstance(self.model.layer_diff_conj_g, torch.nn.Module)
        self.assertIsInstance(self.model.layer_comp, torch.nn.Module)

        # Guess bias
        self.assertIsInstance(self.model.guess_bias, torch.nn.Parameter)

    def test_forward_pass(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)
        agent_type_ids = torch.randint(0, 3, (self.bs,))

        output = self.model(agent_ids, item_embeddings, agent_type_ids)
        self.assertIsInstance(output, HpcirtModelOutput)
        self.assertEqual(output.logits.shape, (self.bs,))
        self.assertEqual(output.difficulty.shape, (self.bs, 1))
        self.assertEqual(output.skill.shape, (self.bs, 5))
        self.assertEqual(output.g.shape, (self.bs, 1))
        self.assertEqual(output.disc_g.shape, (self.bs, 1))
        self.assertEqual(output.disc.shape, (self.bs, 5))
        self.assertEqual(output.diff_conj.shape, (self.bs, 5))
        self.assertEqual(output.diff_conj_g.shape, (self.bs, 1))
        self.assertEqual(output.mu.shape, (self.bs, 1))

    def test_mu_bounds(self):
        """μ should be in [0, 1] since it passes through sigmoid."""
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        output = self.model(agent_ids, item_embeddings)
        self.assertTrue((output.mu >= 0).all())
        self.assertTrue((output.mu <= 1).all())

    def test_disc_positive(self):
        """Discrimination parameters should be non-negative (abs applied)."""
        item_embeddings = torch.randn(self.bs, 32)
        chars = self.model.compute_item_characteristics(item_embeddings)
        self.assertTrue((chars["disc_g"] >= 0).all())
        self.assertTrue((chars["disc"] >= 0).all())

    def test_compute_agent_skills(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        agent_type_ids = torch.randint(0, 3, (self.bs,))

        skills = self.model.compute_agent_skills(agent_ids, agent_type_ids)
        self.assertIn("g", skills)
        self.assertIn("theta", skills)
        self.assertEqual(skills["g"].shape, (self.bs, 1))
        self.assertEqual(skills["theta"].shape, (self.bs, 5))

    def test_compute_item_characteristics(self):
        item_embeddings = torch.randn(self.bs, 32)
        chars = self.model.compute_item_characteristics(item_embeddings)

        expected_keys = {"disc_g", "disc", "difficulty", "diff_conj", "diff_conj_g", "mu"}
        self.assertEqual(set(chars.keys()), expected_keys)

        self.assertEqual(chars["disc_g"].shape, (self.bs, 1))
        self.assertEqual(chars["disc"].shape, (self.bs, 5))
        self.assertEqual(chars["difficulty"].shape, (self.bs, 1))
        self.assertEqual(chars["diff_conj"].shape, (self.bs, 5))
        self.assertEqual(chars["diff_conj_g"].shape, (self.bs, 1))
        self.assertEqual(chars["mu"].shape, (self.bs, 1))

    def test_bifactor_reg_loss(self):
        agent_ids = torch.randint(0, 10, (self.bs,))
        loss = self.model.compute_bifactor_reg_loss(agent_ids)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)

    def test_response_function_compensatory_limit(self):
        """When μ → 0, the model should behave like compensatory MIRT."""
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        # Force μ to 0 by setting compensation layer bias very negative
        with torch.no_grad():
            if isinstance(self.model.layer_comp, torch.nn.Linear):
                self.model.layer_comp.bias.data.fill_(-20.0)
                self.model.layer_comp.weight.data.zero_()
            else:
                for module in reversed(list(self.model.layer_comp.modules())):
                    if isinstance(module, torch.nn.Linear):
                        module.bias.data.fill_(-20.0)
                        module.weight.data.zero_()
                        break

        output = self.model(agent_ids, item_embeddings)
        # Should produce finite logits
        self.assertTrue(torch.isfinite(output.logits).all())
        # μ should be near 0
        self.assertTrue((output.mu < 0.01).all())

    def test_response_function_conjunctive_limit(self):
        """When μ → 1, the model should behave like conjunctive product model."""
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        # Force μ to 1 by setting compensation layer bias very positive
        with torch.no_grad():
            if isinstance(self.model.layer_comp, torch.nn.Linear):
                self.model.layer_comp.bias.data.fill_(20.0)
                self.model.layer_comp.weight.data.zero_()
            else:
                for module in reversed(list(self.model.layer_comp.modules())):
                    if isinstance(module, torch.nn.Linear):
                        module.bias.data.fill_(20.0)
                        module.weight.data.zero_()
                        break

        output = self.model(agent_ids, item_embeddings)
        # Should produce finite logits
        self.assertTrue(torch.isfinite(output.logits).all())
        # μ should be near 1
        self.assertTrue((output.mu > 0.99).all())

    def test_numerical_stability(self):
        """Model should not produce NaN with extreme inputs."""
        agent_ids = torch.randint(0, 10, (self.bs,))

        # Large embeddings
        item_embeddings = torch.randn(self.bs, 32) * 10.0
        output = self.model(agent_ids, item_embeddings)
        self.assertTrue(torch.isfinite(output.logits).all(), "NaN with large inputs")

        # Small embeddings
        item_embeddings = torch.randn(self.bs, 32) * 0.001
        output = self.model(agent_ids, item_embeddings)
        self.assertTrue(torch.isfinite(output.logits).all(), "NaN with small inputs")

    def test_gradient_flow(self):
        """Gradients should flow through the entire model."""
        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)
        labels = torch.randint(0, 2, (self.bs,)).float()

        output = self.model(agent_ids, item_embeddings)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, labels)
        # Include bifactor reg to ensure lambda_loading gets gradients
        loss = loss + 1e-4 * self.model.compute_bifactor_reg_loss(agent_ids)
        loss.backward()

        # Check gradients exist on key parameters
        self.assertIsNotNone(self.model.g_embeddings.weight.grad)
        self.assertIsNotNone(self.model.theta_embeddings.weight.grad)
        self.assertIsNotNone(self.model.lambda_loading.grad)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "ckpt")

            self.model.save_pretrained(ckpt_dir)
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, "model.pt")))

            loaded_model = HpcirtModel.load_pretrained(ckpt_dir)
            self.assertIsInstance(loaded_model, HpcirtModel)

            for param1, param2 in zip(
                self.model.parameters(), loaded_model.parameters()
            ):
                self.assertTrue(torch.equal(param1, param2))

    def test_mlp_mode(self):
        """Test model with MLP heads instead of linear."""
        config = HpcirtConfig(
            n_agents=10,
            n_agent_types=3,
            n_dim=5,
            n_dim_item_embed=32,
            disc_mode="mlp",
            diff_mode="mlp",
            conj_diff_mode="mlp",
            comp_mode="mlp",
            n_hidden=64,
        )
        model = HpcirtModel(config=config)

        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        output = model(agent_ids, item_embeddings)
        self.assertEqual(output.logits.shape, (self.bs,))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_agent_type_required_when_configured(self):
        """Should raise if agent types are expected but not provided."""
        config = HpcirtConfig(
            n_agents=10,
            n_agent_types=3,
            n_dim=5,
            n_dim_item_embed=32,
            fit_agent_type_embeddings=True,
        )
        model = HpcirtModel(config=config)

        agent_ids = torch.randint(0, 10, (self.bs,))
        item_embeddings = torch.randn(self.bs, 32)

        with self.assertRaises(ValueError):
            model(agent_ids, item_embeddings)

    def test_config_arch(self):
        self.assertEqual(self.config.arch, "hpcirt")


if __name__ == "__main__":
    unittest.main()
