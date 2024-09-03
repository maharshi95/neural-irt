import argparse
import os
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor, nn

from neural_irt.configs.common import IrtModelConfig, TrainerConfig
from neural_irt.data.indexers import AgentIndexer
from neural_irt.modelling.caimira import CaimiraModel, CaimiraModelOutput
from neural_irt.modelling.configs import CaimiraConfig, NeuralMIRTConfig


def init_model(config: IrtModelConfig) -> nn.Module:
    if isinstance(config, CaimiraConfig):
        return CaimiraModel(config)
    elif isinstance(config, NeuralMIRTConfig):
        raise NotImplementedError("MIRT model not implemented yet.")
    else:
        raise ValueError(f"Unknown model config: {config}")


class LITModule(pl.LightningModule):
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_or_config: IrtModelConfig | nn.Module,
        val_dataloader_names: list[str] | None = None,
    ):
        super().__init__()

        if isinstance(model_or_config, IrtModelConfig):
            self.model = init_model(model_or_config)
        elif isinstance(model_or_config, nn.Module):
            self.model = model_or_config
        else:
            raise ValueError(
                f"Invalid type for model_or_config: {type(model_or_config)}"
            )
        # self.agent_indexer = agent_indexer
        self.trainer_config = trainer_config
        self.model_config = self.model.config
        self.val_dataloader_names = val_dataloader_names

        self.save_hyperparameters(argparse.Namespace(**trainer_config.model_dump()))

    def forward(self, *args, **kwargs) -> CaimiraModelOutput:
        return self.model.forward(*args, **kwargs)

    def compute_loss(
        self, outputs: CaimiraModelOutput, labels: Tensor
    ) -> dict[str, Tensor]:
        # batch: (subjects, items, labels)
        # return: dict

        loss_ce = F.binary_cross_entropy_with_logits(outputs.logits, labels)
        loss_reg_skill = self.hparams.c_reg_skill * outputs.skill.abs().sum()
        loss_reg_diff = self.hparams.c_reg_difficulty * outputs.difficulty.abs().sum()
        loss_reg = loss_reg_skill + loss_reg_diff
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
        train_metrics = self.compute_loss(outputs, labels)
        with torch.no_grad():
            preds = (outputs.logits > 0).float()
            acc = (preds == labels).float().mean()
        train_metrics["acc"] = acc
        return train_metrics

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        for key, value in outputs.items():
            prog_bar = key == "acc"
            self.log(
                f"train_{key}",
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

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # batch: (subjects, items, labels)
        # return: dict
        labels = batch.pop("labels")
        outputs = self.forward(**batch)
        metrics = self.compute_loss(outputs, labels)

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
                f"val_{key}_{tag}",
                value,
                logger=True,
                add_dataloader_idx=False,
                prog_bar=prog_bar,
            )

    def compute_qualitative_metrics(self):
        metrics = {}

        # [n_items, n_dim]
        rel = self.model.get_relevance()

        rel_std = rel.std(0)

        for i in range(rel.shape[1]):
            metrics[f"rel_std_{i}"] = rel_std[i].item()

        # Create clusters such that if rel[i, j] > 0.5, then i is assigned to cluster (j+1), else to cluster (N+1)
        rel_clusters = (
            torch.where(
                (rel > 0.5).cpu(),
                torch.arange(1, rel.shape[1] + 1),
                rel.shape[1] + 1,
            )
            .min(dim=-1)
            .values
        )

        for i in range(rel.shape[1] + 1):
            metrics[f"cluster_size_rel_{i+1}"] = float(
                (rel_clusters == i + 1).sum().item()
            )

        return metrics

    # def on_validation_epoch_end(self):
    # compute_metrics = self.compute_qualitative_metrics()
    # for name, value in compute_metrics.items():
    #     self.log(name, value, logger=True, add_dataloader_idx=False)

    def predict_step(self, batch, batch_idx):
        # batch: (subjects, items)
        # return: dict
        subjects, items = batch
        logits = self.forward(subjects, items)
        return logits

    def save_checkpoint(self, filepath):
        # save model
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state_dict = self.state_dict()
        state_dict["config.model"] = self.model_config
        state_dict["config.trainer"] = self.trainer_config
        # state_dict["agent_indexer"] = self.agent_indexer.state_dict()
        torch.save(state_dict, filepath)

    def load_checkpoint(self, filepath):
        # load model
        state_dict = torch.load(filepath)
        state_dict.pop("config.model")
        state_dict.pop("config.trainer")
        # agent_indexer = state_dict.pop("agent_indexer")
        self.load_state_dict(state_dict)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        # load model
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        config = state_dict.pop("config.model")
        trainer_config = state_dict.pop("config.trainer")
        agent_indexer = state_dict.pop("agent_indexer")
        model = cls(config, trainer_config, agent_indexer)
        model.load_state_dict(state_dict, strict=False)

        return model
