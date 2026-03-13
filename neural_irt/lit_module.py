import argparse
import os
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from neural_irt.configs.common import IrtModelConfig, TrainerConfig
from neural_irt.data.indexers import AgentIndexer
from neural_irt.modeling.base_models import BaseIrtModel, IrtModelOutput
from neural_irt.modeling.caimira import CaimiraModel
from neural_irt.modeling.configs import CaimiraConfig, HpcirtConfig, MirtConfig
from neural_irt.modeling.hpcirt import HpcirtModel
from neural_irt.modeling.mirt import MirtModel
from neural_irt.utils import config_utils


def create_model(config: IrtModelConfig) -> BaseIrtModel:
    if isinstance(config, HpcirtConfig):
        return HpcirtModel(config)
    elif isinstance(config, CaimiraConfig):
        return CaimiraModel(config)
    elif isinstance(config, MirtConfig):
        return MirtModel(config)
    else:
        raise ValueError(f"Unknown model config: {config}")


def load_model_pretrained(path: str, device: str = "auto") -> BaseIrtModel:
    """Load a pretrained model by inferring the model type from the saved config."""
    config_path = os.path.join(path, "config.json")
    raw_config = config_utils.load_config(config_path)
    if "mu_prior_logit" in raw_config or "bifactor_reg" in raw_config:
        return HpcirtModel.load_pretrained(path, device=device)
    elif "n_dim_item_embed" in raw_config:
        return CaimiraModel.load_pretrained(path, device=device)
    elif "n_items" in raw_config:
        return MirtModel.load_pretrained(path, device=device)
    else:
        raise ValueError(f"Cannot determine model type from config at {path}")


class IrtLitModule(pl.LightningModule):
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_or_config: IrtModelConfig | BaseIrtModel,
        val_dataloader_names: list[str] | None = None,
        agent_indexer: AgentIndexer | None = None,
    ):
        super().__init__()

        if isinstance(model_or_config, IrtModelConfig):
            self.model = create_model(model_or_config)
        elif isinstance(model_or_config, BaseIrtModel):
            self.model = model_or_config
        else:
            raise ValueError(
                f"Invalid type for model_or_config: {type(model_or_config)}"
            )
        self.trainer_config = trainer_config
        self.model_config = self.model.config
        self.val_dataloader_names = val_dataloader_names
        self.agent_indexer = agent_indexer

        self.save_hyperparameters(argparse.Namespace(**trainer_config.model_dump()))

        self._val_relevances: list[Tensor] = []

    def forward(self, *args, **kwargs) -> IrtModelOutput:
        return self.model.forward(*args, **kwargs)

    def compute_loss(
        self, outputs: IrtModelOutput, labels: Tensor, batch: dict | None = None
    ) -> dict[str, Tensor]:
        loss_ce = F.binary_cross_entropy_with_logits(outputs.logits, labels)

        loss_reg = torch.tensor(0.0, device=labels.device)
        c_reg_skill = getattr(self.hparams, "c_reg_skill", 0.0)
        c_reg_difficulty = getattr(self.hparams, "c_reg_difficulty", 0.0)

        if c_reg_skill:
            loss_reg = loss_reg + c_reg_skill * outputs.skill.abs().sum()
        if c_reg_difficulty:
            loss_reg = loss_reg + c_reg_difficulty * outputs.difficulty.abs().sum()

        # HPCIRT-specific regularization
        if hasattr(outputs, "mu"):
            c_reg_g = getattr(self.hparams, "c_reg_g", 0.0)
            c_reg_mu = getattr(self.hparams, "c_reg_mu", 0.0)
            c_reg_disc = getattr(self.hparams, "c_reg_disc", 0.0)
            c_reg_bifactor = getattr(self.hparams, "c_reg_bifactor", 0.0)

            if c_reg_g:
                loss_reg = loss_reg + c_reg_g * outputs.g.pow(2).sum()
            if c_reg_mu:
                # Bimodal penalty: encourage μ toward 0 or 1
                # Penalty = μ(1-μ), maximized at 0.5, zero at 0 and 1
                loss_reg = loss_reg + c_reg_mu * (outputs.mu * (1 - outputs.mu)).sum()
            if c_reg_disc:
                loss_reg = loss_reg + c_reg_disc * outputs.disc.abs().sum()
                loss_reg = loss_reg + c_reg_disc * outputs.disc_g.abs().sum()
            if c_reg_bifactor and batch is not None:
                from neural_irt.modeling.hpcirt import HpcirtModel

                if isinstance(self.model, HpcirtModel) and self.model.config.bifactor_reg:
                    loss_reg = loss_reg + c_reg_bifactor * self.model.compute_bifactor_reg_loss(
                        batch["agent_ids"]
                    )

        loss = loss_ce + loss_reg
        return {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_reg": loss_reg,
        }

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.freeze_bias_after:
            logger.info("Freezing bias parameters at epoch %d", self.current_epoch)
            if self.hparams.fit_guess_bias:
                self.guess_bias.requires_grad = False
                logger.info("Freezing guess bias parameter")

        if self.current_epoch == self.hparams.second_optimizer_start_epoch:
            logger.info("Starting second optimizer at epoch %d", self.current_epoch)

        # Switch to SGD with momentum
        if self.current_epoch == self.hparams.second_optimizer_start_epoch:
            if self.hparams.second_optimizer == "SGD":
                self.optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.hparams.second_learning_rate,
                    momentum=0.9,
                )
                self.trainer.optimizers = [self.optimizer]
            else:
                raise ValueError(
                    f"Optimizer not supported as second optimizer: {self.hparams.second_optimizer}"
                )

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        outputs = self.forward(**batch)
        train_metrics = self.compute_loss(outputs, labels, batch=batch)
        with torch.no_grad():
            preds = (outputs.logits > 0).float()
            acc = (preds == labels).float().mean()
        train_metrics["acc"] = acc
        return train_metrics

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        for key, value in outputs.items():
            prog_bar = key == "acc"
            self.log(
                f"train/{key}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=prog_bar,
            )

        return outputs

    def configure_optimizers(self):
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == "RMSprop":
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}")
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=5e-3,
            max_lr=1e-2,
            step_size_up=1000,
            cycle_momentum=False,
        )
        logger.info(f"Training with optimizer: {optimizer}")
        for name, param in self.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name}: {param.shape}")
        if self.hparams.cyclic_lr:
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]

    def on_validation_epoch_start(self) -> None:
        self._val_relevances = []

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        labels = batch.pop("labels")
        outputs = self.forward(**batch)
        metrics = self.compute_loss(outputs, labels, batch=batch)

        preds = (outputs.logits > 0).float()
        metrics["acc"] = (preds == labels).float().mean()

        recall_0 = (preds[labels == 0] == 0).float().mean()
        recall_1 = (preds[labels == 1] == 1).float().mean()
        metrics["min_recall"] = torch.min(recall_0, recall_1)

        tag = (
            self.val_dataloader_names[dataloader_idx]
            if self.val_dataloader_names
            and dataloader_idx < len(self.val_dataloader_names)
            else f"{dataloader_idx:02d}"
        )
        for key, value in metrics.items():
            prog_bar = key == "acc"
            self.log(
                f"{tag}/{key}",
                value,
                logger=True,
                add_dataloader_idx=True,
                prog_bar=prog_bar,
            )

        if hasattr(outputs, "relevance"):
            self._val_relevances.append(outputs.relevance.detach().cpu())

    def compute_qualitative_metrics(self, rel: Tensor) -> dict[str, float]:
        """Compute qualitative metrics from relevance weights.

        Analyzes the distribution of relevance weights across latent dimensions:
        - Per-dimension standard deviation of relevance weights
        - Cluster assignments based on dominant relevance (threshold > 0.5)

        Args:
            rel: Relevance tensor of shape (n_samples, n_dim), where each row
                 is a probability distribution over latent dimensions.

        Returns:
            Dictionary mapping metric names to float values.
        """
        metrics = {}

        # Per-dimension standard deviation of relevance across items
        rel_std = rel.std(0)
        for i in range(rel.shape[1]):
            metrics[f"rel_std_{i}"] = rel_std[i].item()

        # Cluster items by dominant relevance dimension.
        # If rel[i, j] > 0.5, item i is assigned to cluster (j+1).
        # Items with no dominant dimension go to cluster (n_dim+1).
        n_dim = rel.shape[1]
        rel_clusters = (
            torch.where(
                rel > 0.5,
                torch.arange(1, n_dim + 1),
                n_dim + 1,
            )
            .min(dim=-1)
            .values
        )

        for i in range(n_dim + 1):
            metrics[f"cluster_size_rel_{i + 1}"] = float(
                (rel_clusters == i + 1).sum().item()
            )

        return metrics

    def on_validation_epoch_end(self) -> None:
        if self._val_relevances:
            all_rel = torch.cat(self._val_relevances, dim=0)
            qual_metrics = self.compute_qualitative_metrics(all_rel)
            for name, value in qual_metrics.items():
                self.log(name, value, logger=True, add_dataloader_idx=False)
        self._val_relevances = []

    def predict_step(self, batch, batch_idx):
        logits = self.forward(**batch)
        return logits

    def save_checkpoint(self, dirpath, weights_only=False):
        # save model
        os.makedirs(dirpath, exist_ok=True)
        self.model.save_pretrained(dirpath)
        if not weights_only:
            config_utils.save_config(
                self.trainer_config.model_dump(),
                os.path.join(dirpath, "trainer.json"),
            )
            if self.agent_indexer:
                self.agent_indexer.save_to_disk(dirpath)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        model = load_model_pretrained(checkpoint_path, device=map_location)

        # Use appropriate trainer config class based on model type
        from neural_irt.configs.caimira import TrainerConfig as CaimiraTrainerConfig
        from neural_irt.configs.hpcirt import TrainerConfig as HpcirtTrainerConfig

        trainer_config_path = os.path.join(checkpoint_path, "trainer.json")
        trainer_config_dict = config_utils.load_config(trainer_config_path)
        if isinstance(model.config, HpcirtConfig):
            trainer_config = HpcirtTrainerConfig(**trainer_config_dict)
        elif isinstance(model.config, CaimiraConfig):
            trainer_config = CaimiraTrainerConfig(**trainer_config_dict)
        else:
            trainer_config = TrainerConfig(**trainer_config_dict)

        agent_indexer = None
        if AgentIndexer.exists_on_disk(checkpoint_path):
            agent_indexer = AgentIndexer.load_from_disk(checkpoint_path)
        return cls(
            trainer_config=trainer_config,
            model_or_config=model,
            agent_indexer=agent_indexer,
        )
