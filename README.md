# neural-irt

A PyTorch-based framework for training and evaluating Item Response Theory (IRT) based models, including the Caimira model.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Alternatively, you can install the package using `setup.py`:

```bash
python setup.py install
```

## Quick Start

To train a neural-irt model, follow these steps:

1. Create a configuration file
2. Run the training script

### 1. Create a Configuration File

Create a new YAML configuration file in the `configs/` directory. For example, `configs/caimira_configs/my_caimira_config.yaml`:

```yaml
data:
    train_set:
        queries: "path/to/train_queries.jsonl"
        agents: "path/to/train_agents.jsonl"
        responses: "path/to/train_responses.jsonl"
    val_sets:
        val:
            queries: "path/to/val_queries.jsonl"
            responses: "path/to/val_responses.jsonl"
        test_set:
            queries: "path/to/test_queries.jsonl"
            responses: "path/to/test_responses.jsonl"
    question_input_format: "embedding"
    agent_input_format: "id"
    query_embeddings_path: "path/to/query_embeddings.pt"
model:
    n_agents: 1000
    n_agent_types: 5
    n_dim: 32
    n_dim_item_embed: 768
    rel_mode: "linear"
    dif_mode: "linear"
    fit_guess_bias: false
    fit_agent_type_embeddings: true
    rel_temperature: 0.5
trainer:
    c_reg_skill: 1e-6
    c_reg_difficulty: 1e-6
    c_reg_relevance: 1e-6
    batch_size: 256
    max_epochs: 50
    optimizer: "Adam"
    learning_rate: 1e-3
wandb:
    project: "my-neural-irt-project"
    save_dir: "./wandb_logs"
```

You can also refer to an example config file [test-run.yaml](configs/test-run.yaml)

### 2. Run the Training Script

Use the `neural_irt.train` module to start training. You can override configuration values using command-line arguments:

```bash
python -m neural_irt.train --config-paths configs/caimira_configs/my_caimira_config.yaml \
--trainer.max_epochs=10 \
--trainer.batch_size=512 \
```

This command will:
- Load the configuration from `my_caimira_config.yaml`
- Override the maximum number of epochs to 10
- Set the batch size to 512

## Configuration

Key configuration options include:

- `data`: Specifies paths to training and validation data
- `model`: Defines the model architecture and parameters
- `trainer`: Sets training hyperparameters
- `wandb`: Configures Weights & Biases logging

Refer to the configuration files in [`configs/`](configs/) or the config classes in [`neural_irt/configs`](neural_irt/configs) for more detailed examples.

The main configuration classes are:

- `CaimiraConfig`: Configuration for the Caimira model.
- `TrainerConfig`: Configuration for the training process.
- `RunConfig`: Configuration for a complete run, including model and trainer configurations.

## Monitoring Training

The training script uses Weights & Biases (wandb) for experiment tracking. You can monitor your training runs by logging into your wandb account and viewing the project specified in the configuration.


## Evaluation

To evaluate a trained model, you can use the `CaimiraInferenceModel` class:

```python
from neural_irt.evals.models import CaimiraInferenceModel

# Load pretrained model
model = CaimiraInferenceModel.load_pretrained("path/to/checkpoint", device="cpu")

# Compute agent skills
agent_names = ["agent_1", "agent_2", "agent_3"]
skills = model.compute_agent_skills(agent_names)
print("Agent skills:", skills)
```

## Testing

Unit tests are located in the `tests` directory. To run the tests, use:

```bash
bash run_tests.sh
```

Example test case:

```python
import unittest
from neural_irt.modelling.caimira import CaimiraConfig, CaimiraModel

class TestCaimiraModel(unittest.TestCase):
    def setUp(self):
        self.config = CaimiraConfig(n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32)
        self.model = CaimiraModel(config=self.config)

    def test_forward(self):
        batch_size = 64
        agent_ids = torch.randint(0, 10, (batch_size,))
        item_embeddings = torch.randn(batch_size, 32)
        agent_type_ids = torch.randint(0, 3, (batch_size,))
        output = self.model(agent_ids, item_embeddings, agent_type_ids)
        self.assertEqual(output.logits.shape, (batch_size,))

if __name__ == "__main__":
    unittest.main()
```

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## License

This project is licensed under the GPLv2 License. See the [LICENSE](LICENSE) file for details.