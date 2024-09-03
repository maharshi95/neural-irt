from typing import Any, Optional

from pydantic import BaseModel

from neural_irt.modeling.configs import IrtModelConfig


class DatasetConfig(BaseModel):
    queries: str
    agents: str
    responses: str


class DataConfig(BaseModel):
    train_set: DatasetConfig
    val_set: Optional[DatasetConfig]
    val_sets: dict[str, DatasetConfig]
    question_input_format: str = "id"
    agent_input_format: str = "id"
    agent_indexer_path: Optional[str] = None
    query_indexer_path: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        # Make sure only one of val_set and val_sets is set
        if not (bool(self.val_set) ^ bool(self.val_sets)):
            raise ValueError("Exactly one of val_set and val_sets must be set")
        if self.val_set:
            self.val_sets = {"val": self.val_set}
            self.val_set = None


class TrainerConfig(BaseModel):
    # Train time
    max_epochs: int = 100
    max_steps: Optional[int] = None
    sampler: Optional[str] = None
    batch_size: int = 32

    # Optimizer
    optimizer: str = "Adam"  # [Adam, RMSprop, SGD]
    learning_rate: float = 1e-3
    cyclic_lr: bool = False

    second_optimizer: str = "SGD"
    second_learning_rate: float = 5e-4
    second_optimizer_start_epoch: Optional[int] = 75

    freeze_bias_after: Optional[int] = None

    ckpt_savedir: str = "./checkpoints/irt"


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str
    save_dir: str
    entity: Optional[str] = None


class RunConfig(BaseModel):
    run_name: Optional[str] = None
    extra_run_name_tag: Optional[str] = None
    model: IrtModelConfig
    trainer: TrainerConfig
    data: DataConfig
    wandb: Optional[WandbConfig] = None


# Rest of the code remains the same
