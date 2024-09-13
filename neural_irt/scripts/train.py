import argparse
import os
from collections import Counter
from typing import Any, Optional, Sequence

from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler
from rich.traceback import install
from rich_argparse import RichHelpFormatter
from torch.utils import data as torch_data

from neural_irt.configs.caimira import RunConfig
from neural_irt.configs.common import DataConfig
from neural_irt.data import collators, datasets
from neural_irt.data.indexers import AgentIndexer
from neural_irt.train_model import LITModule
from neural_irt.utils import config_utils, parser_utils

install(show_locals=False, width=120, extra_lines=2, code_width=90)


logger.configure(
    handlers=[
        {
            "sink": RichHandler(markup=True, rich_tracebacks=True),
            "format": "[green]{name}[/green]:{function}:{line} - {message}",
        }
    ]
)


def save_checkpoint(
    lit_model: LITModule, agent_indexer: AgentIndexer, config: RunConfig, path: str
):
    os.makedirs(path, exist_ok=True)
    lit_model.model.save_pretrained(path)
    agent_indexer.save_to_disk(path)


def make_run_name(config: RunConfig) -> str:
    model_config = config.model
    name_str = f"{model_config.arch}-{model_config.n_dim}-dim"

    name_str += f"_diff-{args.diff_mode}"

    name_str += f"_imp-{args.imp_mode}"
    if args.sparse_imp:
        name_str += "-sparse"
    if args.imp_mode == "topics":
        topic_model_name = "lda" if "lda" in args.topic_model_datapath else "bert"
        name_str += f"_{topic_model_name}"
    elif args.imp_mode == "kernel":
        emb_model = (
            args.embeddings_filepath.split("/")[-1]
            .split("-embeddings-")[-1]
            .split(".")[0]
        )
        words = (
            args.embeddings_filepath.split("/")[-1]
            .split("-embeddings-")[0]
            .removeprefix("pb-")
        ).split("-")
        if words[-1] == "cross":
            words[-1] = "x"
        initials = "".join([w[0] for w in words])
        name_str += f"_{initials}-emb-{emb_model}"

    fit_param_names = []
    if model_config.fit_guess_bias:
        fit_param_names.append("gbias")
    if model_config.fit_agent_type_embeddings:
        fit_param_names.append("atype")
    name_str += f"_fit-{'-'.join(fit_param_names)}"
    if model_config.characteristics_bounder is not None:
        name_str += "_" + model_config.characteristics_bounder.to_string

    trainer_config = config.trainer
    if trainer_config.freeze_bias_after is not None:
        name_str += f"_freeze-after-{trainer_config.freeze_bias_after}"

    name_str += f"_{trainer_config.optimizer}-lr={trainer_config.learning_rate:.0e}"
    name_str += (
        "_c-reg"
        + f"-skill={trainer_config.c_reg_skill:.0e}".replace("e-0", "e-")
        + f"-diff={trainer_config.c_reg_difficulty:.0e}".replace("e-0", "e-")
        + f"-imp={trainer_config.c_reg_relevance:.0e}".replace("e-0", "e-")
    )

    if trainer_config.sampler is not None:
        name_str += f"_sampler={trainer_config.sampler}"
    return name_str


def create_agent_indexer_from_dataset(
    dataset_or_path: str | Sequence[dict[str, Any]],
) -> AgentIndexer:
    dataset = dataset_or_path
    if isinstance(dataset, str):
        dataset = datasets.load_as_hf_dataset(dataset_or_path)

    agent_ids = [entry["id"] for entry in dataset]
    agent_types = list({entry["agent_type"] for entry in dataset})
    agent_type_map = {entry["id"]: entry["agent_type"] for entry in dataset}
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
    print("# Agents: ", agent_indexer.n_agents)
    print("# Agent Types: ", agent_indexer.n_agent_types)
    train_collator = collators.CaimiraCollator(agent_indexer, is_training=True)
    val_collator = collators.CaimiraCollator(agent_indexer, is_training=False)

    train_dataset = datasets.load_irt_dataset(
        data_config.train_set,
        data_config.question_input_format,
        data_config.agent_input_format,
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
    )
    val_loaders = {}
    for name, val_set in data_config.val_sets.items():
        val_ds = datasets.load_irt_dataset(
            val_set,
            data_config.question_input_format,
            data_config.agent_input_format,
        )
        val_loaders[name] = torch_data.DataLoader(
            val_ds,
            batch_size=train_batch_size,
            shuffle=False,
            collate_fn=val_collator,
        )
    labels_ctr = Counter(e["ruling"] for e in train_dataset)
    random_chance = max(labels_ctr.values()) / len(train_dataset)
    logger.info(f"Random chance Accuracy: {random_chance * 100:.1f}%")

    return train_loader, val_loaders


def main(args: argparse.Namespace) -> None:
    config: RunConfig = config_utils.load_config_from_namespace(args, RunConfig)
    agent_indexer = create_agent_indexer(config)

    exp_name = config.run_name or make_run_name(config)
    logger.info("Experiment Name: " + exp_name)

    train_loader, val_loaders_dict = make_dataloaders(
        config.data, agent_indexer, config.trainer.batch_size
    )
    val_loader_names = list(val_loaders_dict.keys())
    val_loaders = [val_loaders_dict[name] for name in val_loader_names]
    model = LITModule(
        config.trainer, config.model, val_dataloader_names=val_loader_names
    )
    logger.info(f"Model loaded with the following config:\n{config.model}")

    train_logger = None
    if config.wandb and config.wandb.enabled:
        train_logger = WandbLogger(
            project=config.wandb.project,
            name=exp_name,
            save_dir=config.wandb.save_dir,
            entity=config.wandb.entity,
        )

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator="auto",
        logger=train_logger,
        # callbacks=[
        #     pl.callbacks.ModelCheckpoint(
        #         save_top_k=3,
        #         monitor="val_loss",
        #         mode="min",
        #         dirpath="checkpoints",
        #         filename="{epoch}-{val_loss:.2f}",
        #     )
        # ],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loaders)
    model_ckpt_path = f"{config.trainer.ckpt_savedir}/{exp_name}.ckpt"
    trainer_ckpt_path = model_ckpt_path.replace(".ckpt", ".trainer.ckpt")
    model.save_checkpoint(model_ckpt_path)
    trainer.save_checkpoint(trainer_ckpt_path)
    ckpt_dir = f"{config.trainer.ckpt_savedir}/{exp_name}/epoch_{trainer.current_epoch}"
    save_checkpoint(model, agent_indexer, config, ckpt_dir)
    logger.info(f"Saved model to {ckpt_dir}")


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
    logger.info(args)
    main(args)


# %%
