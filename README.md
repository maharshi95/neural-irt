# neural-irt

A PyTorch-based framework for training and evaluating Item Response Theory (IRT) models, including the Caimira model.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
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

Here's a quick example to get you started with training a Caimira model:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from neural_irt.modelling.caimira import CaimiraConfig, CaimiraModel
from neural_irt.configs.caimira import TrainerConfig

# Define configurations
model_config = CaimiraConfig(n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32)
trainer_config = TrainerConfig(max_epochs=10, batch_size=32)

# Initialize model
model = CaimiraModel(config=model_config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Dummy data
agent_ids = torch.randint(0, 10, (1000,))
item_embeddings = torch.randn(1000, 32)
agent_type_ids = torch.randint(0, 3, (1000,))
labels = torch.randint(0, 2, (1000,))

# Create DataLoader
dataset = TensorDataset(agent_ids, item_embeddings, agent_type_ids, labels)
dataloader = DataLoader(dataset, batch_size=trainer_config.batch_size, shuffle=True)

# Training loop
for epoch in range(trainer_config.max_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        agent_ids_batch, item_embeddings_batch, agent_type_ids_batch, labels_batch = batch
        optimizer.zero_grad()
        output = model(agent_ids_batch, item_embeddings_batch, agent_type_ids_batch)
        loss = criterion(output.logits, labels_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{trainer_config.max_epochs}, Loss: {total_loss/len(dataloader)}")
```

## Project Structure

The project is organized as follows:

- `neural_irt/`: Core library code.
  - `modeling/`: Model definitions and related utilities.
  - `configs/`: Configuration classes.
  - `data/`: Data processing and indexing utilities.
  - `evals/`: Evaluation scripts and utilities.
  - `utils/`: Utility functions.
  - `scripts/`: Scripts for various tasks.
  - `tests/`: Unit tests for the project.
- `setup.py`: Installation script.
- `README.md`: Project documentation.

## Configuration

Configuration classes are defined in the `neural_irt/configs` directory. The main configuration classes are:

- `CaimiraConfig`: Configuration for the Caimira model.
- `TrainerConfig`: Configuration for the training process.
- `RunConfig`: Configuration for a complete run, including model and trainer configurations.

Example configuration:

```python
from neural_irt.configs.caimira import CaimiraConfig, TrainerConfig, RunConfig

model_config = CaimiraConfig(n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32)
trainer_config = TrainerConfig(max_epochs=100, batch_size=32)
run_config = RunConfig(model=model_config, trainer=trainer_config)
```

## Training

To train a model, you need to prepare your data and define your configurations. Here's an example of how to train a Caimira model:

```python
from neural_irt.modelling.caimira import CaimiraModel
from neural_irt.configs.caimira import CaimiraConfig, TrainerConfig, RunConfig

# Define configurations
model_config = CaimiraConfig(n_agents=10, n_agent_types=3, n_dim=7, n_dim_item_embed=32)
trainer_config = TrainerConfig(max_epochs=100, batch_size=32)
run_config = RunConfig(model=model_config, trainer=trainer_config)

# Initialize model
model = CaimiraModel(config=model_config)

# Training loop (simplified)
for epoch in range(trainer_config.max_epochs):
    for batch in train_dataloader:
        agent_ids, item_embeddings, agent_type_ids, labels = batch
        output = model(agent_ids, item_embeddings, agent_type_ids)
        loss = compute_loss(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

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
pytest
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