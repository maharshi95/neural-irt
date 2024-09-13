import argparse
import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from typing import Any

import pydantic
from pydantic import BaseModel

from neural_irt.utils import config_utils

base_run_config_yaml = """
exp_name: "experiment_1"

model:
  name_or_path: "bert-base-uncased"
  n_dim: 768
  tokenizer:
    name_or_path: "bert-base-uncased"

data:
  path: "/path/to/your/dataset"
  batch_size: 32
  num_workers: 4
  pin_memory: true
  device: "cuda"

trainer:
  max_epochs: 10
  accelerator: "gpu"
  logger: "tensorboard"
"""

run_config_trainer_patch_yaml = """
exp_name: "wandb_exp"

trainer:
  max_epochs: 7
  accelerator: "tpu"
  logger: "wandb"
"""

model_config_yaml = """
name_or_path: "meta-llama/Llama-3-8b"
n_dim: 2048
tokenizer:
  name_or_path: "meta-llama/Llama-2-8b"
"""

tokenizer_config_yaml = """
name_or_path: "meta-llama/Llama-2-8b"
model_max_length: 1111
"""


class TokenizerConfig(BaseModel):
    name_or_path: str
    model_max_length: int | str = "auto"


class ModelConfig(BaseModel):
    name_or_path: str
    n_dim: int
    tokenizer: TokenizerConfig


class DataConfig(BaseModel):
    path: str
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"


class TrainerConfig(BaseModel):
    max_epochs: int
    accelerator: str
    logger: str


class RunConfig(BaseModel):
    exp_name: str
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig

    def model_post_init(self, __context: Any) -> None:
        # Make sure if exp_name is starting with "exp" then model.n_dim must be 768
        if self.model.name_or_path.startswith("bert-base"):
            assert (
                self.model.n_dim == 768
            ), "For exp_name starting with 'exp', model.n_dim must be 768"


class TestConfigUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base_run_config_path = os.path.join(self.temp_dir, "base_run_config.yaml")
        self.run_config_trainer_patch_path = os.path.join(
            self.temp_dir, "run_config_trainer_patch.yaml"
        )
        self.model_config_path = os.path.join(self.temp_dir, "model_config.yaml")
        self.tokenizer_config_path = os.path.join(
            self.temp_dir, "tokenizer_config.yaml"
        )

        with open(self.base_run_config_path, "w") as f:
            f.write(base_run_config_yaml)
        with open(self.run_config_trainer_patch_path, "w") as f:
            f.write(run_config_trainer_patch_yaml)
        with open(self.model_config_path, "w") as f:
            f.write(model_config_yaml)
        with open(self.tokenizer_config_path, "w") as f:
            f.write(tokenizer_config_yaml)

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_save_config(self):
        config = RunConfig(
            exp_name="test_exp",
            model=ModelConfig(
                name_or_path="test_model",
                n_dim=768,
                tokenizer=TokenizerConfig(name_or_path="test_tokenizer"),
            ),
            data=DataConfig(path="test_data"),
            trainer=TrainerConfig(
                max_epochs=10, accelerator="cpu", logger="tensorboard"
            ),
        )

        # Test saving as YAML
        yaml_path = os.path.join(self.temp_dir, "config.yaml")
        config_utils.save_config(config, yaml_path)
        self.assertTrue(os.path.exists(yaml_path))

        # Test saving as JSON
        json_path = os.path.join(self.temp_dir, "config.json")
        config_utils.save_config(config, json_path)
        self.assertTrue(os.path.exists(json_path))

        # Test unsupported file extension
        with self.assertRaises(ValueError):
            config_utils.save_config(config, os.path.join(self.temp_dir, "config.txt"))

    def test_load_config(self):
        # Create sample config files
        model_config = {
            "name_or_path": "meta-llama/Llama-3-8b",
            "n_dim": 2048,
            "tokenizer": {"name_or_path": "meta-llama/Llama-2-8b"},
        }
        trainer_config = {
            "max_epochs": 7,
            "accelerator": "gpu",
            "logger": "wandb",
        }

        model_path = os.path.join(self.temp_dir, "model_config.yaml")
        trainer_path = os.path.join(self.temp_dir, "trainer_config.yaml")

        with open(model_path, "w") as f:
            config_utils.yaml.dump(model_config, f)
        with open(trainer_path, "w") as f:
            config_utils.yaml.dump(trainer_config, f)

        # Test loading single config
        loaded_model_config = config_utils.load_config(model_path)
        self.assertEqual(loaded_model_config, model_config)

        # Test loading multiple configs
        loaded_configs = config_utils.load_config(model_path, trainer_path)
        expected_config = {**model_config, **trainer_config}
        self.assertEqual(loaded_configs, expected_config)

        # Test loading with Pydantic model
        loaded_model_config = config_utils.load_config(model_path, cls=ModelConfig)
        self.assertIsInstance(loaded_model_config, ModelConfig)
        self.assertEqual(loaded_model_config.name_or_path, "meta-llama/Llama-3-8b")

        # Test loading non-existent file
        with self.assertRaises(FileNotFoundError):
            config_utils.load_config("non_existent.yaml")

    def test_edge_cases(self):
        # Test empty config paths
        with self.assertRaises(ValueError):
            config_utils.load_config()

        # Test loading config with missing required fields
        incomplete_config = {"exp_name": "test"}
        incomplete_path = os.path.join(self.temp_dir, "incomplete.yaml")
        with open(incomplete_path, "w") as f:
            config_utils.yaml.dump(incomplete_config, f)

        with self.assertRaises(ValueError):
            config_utils.load_config(incomplete_path, cls=RunConfig)

        # Test loading config with extra fields
        extra_config = {
            "exp_name": "test",
            "model": {
                "name_or_path": "test",
                "n_dim": 768,
                "tokenizer": {"name_or_path": "test"},
            },
            "data": {"path": "test"},
            "trainer": {
                "max_epochs": 10,
                "accelerator": "cpu",
                "logger": "tensorboard",
            },
            "extra_field": "should be ignored",
        }
        extra_path = os.path.join(self.temp_dir, "extra.yaml")
        with open(extra_path, "w") as f:
            config_utils.yaml.dump(extra_config, f)

        loaded_config = config_utils.load_config(extra_path, cls=RunConfig)
        self.assertIsInstance(loaded_config, RunConfig)
        self.assertFalse(hasattr(loaded_config, "extra_field"))

    def test_load_config_from_multiple_files(self):
        # Test loading config from multiple files
        loaded_config = config_utils.load_config(
            self.base_run_config_path, self.run_config_trainer_patch_path, cls=RunConfig
        )
        self.assertIsInstance(loaded_config, RunConfig)
        self.assertEqual(loaded_config.exp_name, "wandb_exp")
        self.assertEqual(loaded_config.trainer.max_epochs, 7)
        self.assertEqual(loaded_config.trainer.logger, "wandb")
        self.assertEqual(loaded_config.model.name_or_path, "bert-base-uncased")

    def test_load_config_with_overrides(self):
        # Test loading config with overrides
        model_patch_yaml = dedent(
            """
            model:
              name_or_path: "meta-llama/Llama-3-8b"
              n_dim: 2048
              tokenizer:
                name_or_path: "meta-llama/Llama-2-8b"
            """
        )
        model_patch_path = os.path.join(self.temp_dir, "model_patch.yaml")
        with open(model_patch_path, "w") as f:
            f.write(model_patch_yaml)

        loaded_config = config_utils.load_config(
            self.base_run_config_path, model_patch_path, cls=RunConfig
        )
        self.assertIsInstance(loaded_config, RunConfig)
        self.assertEqual(loaded_config.model.name_or_path, "meta-llama/Llama-3-8b")
        self.assertEqual(loaded_config.model.n_dim, 2048)
        self.assertEqual(
            loaded_config.model.tokenizer.name_or_path, "meta-llama/Llama-2-8b"
        )

    def test_load_config_with_partial_override(self):
        # Test loading config with partial override
        partial_override = """
        trainer:
          max_epochs: 5
        """
        partial_override_path = os.path.join(self.temp_dir, "partial_override.yaml")
        with open(partial_override_path, "w") as f:
            f.write(partial_override)

        loaded_config = config_utils.load_config(
            self.base_run_config_path,
            self.run_config_trainer_patch_path,
            partial_override_path,
            cls=RunConfig,
        )
        self.assertIsInstance(loaded_config, RunConfig)
        self.assertEqual(loaded_config.exp_name, "wandb_exp")
        self.assertEqual(loaded_config.trainer.max_epochs, 5)
        self.assertEqual(loaded_config.trainer.accelerator, "tpu")
        self.assertEqual(loaded_config.trainer.logger, "wandb")
        self.assertEqual(loaded_config.model.name_or_path, "bert-base-uncased")
        self.assertEqual(loaded_config.model.n_dim, 768)
        self.assertEqual(
            loaded_config.model.tokenizer.name_or_path, "bert-base-uncased"
        )

    def test_load_config_with_model_post_init(self):
        # Test loading config that triggers model_post_init
        config_with_exp = dedent("""
        exp_name: "exp_test"
        model:
          name_or_path: "test_model"
          n_dim: 768
          tokenizer:
            name_or_path: "test_tokenizer"
        data:
          path: "test_data"
        trainer:
          max_epochs: 10
          accelerator: "cpu"
          logger: "tensorboard"
        """)
        config_with_exp_path = os.path.join(self.temp_dir, "config_with_exp.yaml")
        with open(config_with_exp_path, "w") as f:
            f.write(config_with_exp)

        loaded_config = config_utils.load_config(config_with_exp_path, cls=RunConfig)
        self.assertIsInstance(loaded_config, RunConfig)
        self.assertEqual(loaded_config.exp_name, "exp_test")
        self.assertEqual(loaded_config.model.n_dim, 768)

        # Test that assertion is raised when n_dim is not 768
        config_with_wrong_n_dim = """
        exp_name: "exp_test"
        model:
          name_or_path: "bert-base-uncased"
          n_dim: 512
          tokenizer:
            name_or_path: "test_tokenizer"
        data:
          path: "test_data"
        trainer:
          max_epochs: 10
          accelerator: "cpu"
          logger: "tensorboard"
        """
        config_with_wrong_n_dim_path = os.path.join(
            self.temp_dir, "config_with_wrong_n_dim.yaml"
        )
        with open(config_with_wrong_n_dim_path, "w") as f:
            f.write(config_with_wrong_n_dim)

        with self.assertRaises(pydantic.ValidationError):
            config_utils.load_config(config_with_wrong_n_dim_path, cls=RunConfig)


if __name__ == "__main__":
    unittest.main()
