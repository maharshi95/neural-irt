import argparse
import os
from collections import Counter
from typing import Any, Optional, Sequence

from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler
from rich.traceback import install
from rich_argparse import RichHelpFormatter
from torch.utils import data as torch_data
from rich import print as rprint
import wandb
from neural_irt.configs.caimira import RunConfig
from neural_irt.configs.common import DataConfig
from neural_irt.data import collators, datasets
from neural_irt.data.indexers import AgentIndexer
from neural_irt.lit_module import IrtLitModule
from neural_irt.utils import config_utils, parser_utils

install(show_locals=False, width=120, extra_lines=2)


logger.configure(
    handlers=[
        {
            "sink": RichHandler(markup=True, rich_tracebacks=True),
            "format": "[green]{name}[/green]:{function}:{line} - {message}",
        }
    ]
)


def make_run_name(config: RunConfig) -> str:
    model_config = config.model

    def get_embedding_tag(filepath: str) -> str:
        emb_filepath = filepath
        emb_model = emb_filepath.split("/")[-1].split("-embeddings-")[-1].split(".")[0]
        words = (
            emb_filepath.split("/")[-1].split("-embeddings-")[0].removeprefix("pb-")
        ).split("-")
        if words[-1] == "cross":
            words[-1] = "x"
        initials = "".join([w[0] for w in words])
        return f"_{initials}-emb-{emb_model}"

    def get_params_tag():
        param_names = []
        if model_config.fit_guess_bias:
            param_names.append("gbias")
        if model_config.fit_agent_type_embeddings:
            param_names.append("atype")
        tag = f"_fit-{'-'.join(param_names)}" if param_names else ""
        if model_config.characteristics_bounder is not None:
            tag += f"_{model_config.characteristics_bounder.to_string}"
        return tag

    def get_trainer_tag():
        cfg = config.trainer
        trainer_tags = []
        if cfg.freeze_bias_after is not None:
            trainer_tags.append(f"_freeze-after-{cfg.freeze_bias_after}")

        trainer_tags.append(f"_{cfg.optimizer}-lr={cfg.learning_rate:.0e}")
        trainer_tags.append(
            "c-reg"
            + f"-skill={cfg.c_reg_skill:.0e}".replace("e-0", "e-")
            + f"-diff={cfg.c_reg_difficulty:.0e}".replace("e-0", "e-")
            + f"-imp={cfg.c_reg_relevance:.0e}".replace("e-0", "e-")
        )

        if cfg.sampler is not None:
            trainer_tags.append(f"_sampler={cfg.sampler}")
        return "".join(trainer_tags)

    model_arch_tag = f"{model_config.arch}-{model_config.n_dim}-dim"
    emb_tag = get_embedding_tag(config.data.query_embeddings_path)
    params_tag = get_params_tag()
    trainer_tag = get_trainer_tag()

    return f"{model_arch_tag}{emb_tag}{params_tag}{trainer_tag}"


def create_agent_indexer_from_dataset(
    dataset_or_path: str | Sequence[dict[str, Any]],
) -> AgentIndexer:
    dataset = dataset_or_path
    if isinstance(dataset, str):
        dataset = datasets.load_as_hf_dataset(dataset_or_path)

    agent_ids = [entry["id"] for entry in dataset]
    agent_types = list({entry["type"] for entry in dataset})
    agent_type_map = {entry["id"]: entry["type"] for entry in dataset}
    return AgentIndexer(agent_ids, agent_types, agent_type_map)


def create_agent_indexer(config: RunConfig) -> AgentIndexer:
    if config.data.agent_indexer_path:
        agent_indexer = AgentIndexer.load_from_disk(config.data.agent_indexer_path)
    else:
        agent_indexer = create_agent_indexer_from_dataset(config.data.train_set.agents)

    # If the model config is not set, set it to the agent indexer with a warning
    if not config.model.n_agents:
        logger.warning(
            "n_agents not set in model config, "
            f"setting to agent_indexer.n_agents: {agent_indexer.n_agents}"
        )
        config.model.n_agents = agent_indexer.n_agents

    if not config.model.n_agent_types:
        logger.warning(
            "n_agent_types not set in model config, "
            f"setting to agent_indexer.n_agent_types: {agent_indexer.n_agent_types}"
        )
        config.model.n_agent_types = agent_indexer.n_agent_types

    return agent_indexer


def make_dataloaders(
    data_config: DataConfig, agent_indexer: AgentIndexer, train_batch_size: int
):
    train_collator = collators.CaimiraCollator(agent_indexer, is_training=True)
    val_collator = collators.CaimiraCollator(agent_indexer, is_training=False)

    train_dataset = datasets.load_irt_dataset(
        data_config.train_set,
        data_config.question_input_format,
        data_config.agent_input_format,
        data_config.query_embeddings_path,
    )

    # if data_config.trainer.sampler == "weighted":
    #     # TODO: Implement this
    #     raise NotImplementedError("Weighted sampler not implemented")
    #     logger.info("Using Weighted Sampler.")

    #     # Gather indices from train_dataset that correspond to subject types
    #     ds_indices = {
    #         i
    #         for i, e in enumerate(train_dataset)
    #         if "human" in agent_indexer.get_agent_type(e["agent_id"])
    #     }
    #     w_humans = 3.0
    #     w_ai = 1.0
    #     weights = [
    #         w_humans if i in ds_indices else w_ai for i in range(len(train_dataset))
    #     ]
    #     sampler = torch_data.WeightedRandomSampler(weights, len(weights))

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=1,
    )
    val_loaders = {}
    for name, val_set in data_config.val_sets.items():
        val_ds = datasets.load_irt_dataset(
            val_set,
            data_config.question_input_format,
            data_config.agent_input_format,
            data_config.query_embeddings_path,
        )
        val_loaders[name] = torch_data.DataLoader(
            val_ds,
            batch_size=train_batch_size,
            shuffle=False,
            collate_fn=val_collator,
            num_workers=1,
        )
    labels_ctr = Counter(e["ruling"] for e in train_dataset)
    random_chance = max(labels_ctr.values()) / len(train_dataset)
    logger.info(f"Random chance Accuracy: {random_chance * 100:.1f}%")

    return train_loader, val_loaders


class CaimiraTrainer(Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        checkpoint_dir = filepath.rstrip(".ckpt")
        logger.info(f"Saving checkpoint to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_checkpoint(checkpoint_dir, weights_only=weights_only)


def main(args: argparse.Namespace) -> None:
    config: RunConfig = config_utils.load_config_from_namespace(args, RunConfig)
    logger.info(f"Loaded config:\n{config}")
    agent_indexer = create_agent_indexer(config)

    run_name = config.run_name or make_run_name(config)
    logger.info("Experiment Name: " + run_name)

    train_loader, val_loaders_dict = make_dataloaders(
        config.data, agent_indexer, config.trainer.batch_size
    )
    val_loader_names = list(val_loaders_dict.keys())
    val_loaders = [val_loaders_dict[name] for name in val_loader_names]
    model = IrtLitModule(
        config.trainer,
        config.model,
        val_dataloader_names=val_loader_names,
        agent_indexer=agent_indexer,
    )
    logger.info(f"Model loaded with the following config:\n{config.model}")

    train_logger = None
    save_dir = os.path.abspath(config.wandb.save_dir)
    if config.wandb and config.wandb.enabled:
        train_logger = WandbLogger(
            project=config.wandb.project,
            name=run_name,
            dir=save_dir,
            entity=config.wandb.entity,
        )

    ckpt_dir = f"{config.trainer.ckpt_savedir}/{run_name}"

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val/acc",
        mode="min",
        dirpath=ckpt_dir,
        auto_insert_metric_name=False,
        filename="epoch={epoch}-loss={val/loss:.2f}",
    )
    checkpoint_callback.FILE_EXTENSION = ""
    trainer = CaimiraTrainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="auto",
        logger=train_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    # model_ckpt_path = f"{config.trainer.ckpt_savedir}/{exp_name}.ckpt"
    # trainer_ckpt_path = model_ckpt_path.replace(".ckpt", ".trainer.ckpt")

    # model.save_checkpoint(model_ckpt_path)
    # trainer.save_checkpoint(trainer_ckpt_path)

    # logger.info(f"Saved model to {model_ckpt_path}")
    logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

    loaded_model = IrtLitModule.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    print(loaded_model)
    print("Run dir:", wandb.run.dir)
    wandb.save()


def add_arguments(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()

    return parser_utils.populate_parser_with_config_args(parser, RunConfig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=RichHelpFormatter, description="Train IRT model."
    )
    parser = add_arguments(parser)
    args = parser.parse_args()
    rprint({k: v for k, v in vars(args).items() if v is not None})
    main(args)
