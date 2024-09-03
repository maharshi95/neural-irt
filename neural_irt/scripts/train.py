import argparse
from collections import Counter
from typing import Any, Optional, Sequence

from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from rich.logging import RichHandler
from rich.traceback import install
from rich_argparse import RichHelpFormatter
from torch.utils import data as torch_data

from neural_irt import config_utils
from neural_irt.configs import CaimiraDatasetConfig, RunConfig
from neural_irt.data import collators, datasets
from neural_irt.data.datasets import load_dataset_hf
from neural_irt.data.indexers import AgentIndexer
from neural_irt.train_model import LITModule

install(show_locals=False, width=120, extra_lines=2, code_width=90)


logger.configure(
    handlers=[
        {
            "sink": RichHandler(markup=True, rich_tracebacks=True),
            "format": "[green]{name}[/green]:{function}:{line} - {message}",
        }
    ]
)


def create_agent_indexer_from_dataset(
    dataset_or_path: str | Sequence[dict[str, Any]],
) -> AgentIndexer:
    dataset = dataset_or_path
    if isinstance(dataset, str):
        dataset = load_dataset_hf(dataset_or_path)

    agent_ids = [entry["id"] for entry in dataset]
    agent_types = list(set([entry["agent_type"] for entry in dataset]))
    agent_type_map = {entry["id"]: entry["agent_type"] for entry in dataset}
    return AgentIndexer(agent_ids, agent_types, agent_type_map)


def make_run_name(config: RunConfig) -> str:
    model_config = config.model.config
    name_str = f"{config.model.arch}-{model_config.n_dim}-dim"

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


def load_dataset(
    dataset_config: CaimiraDatasetConfig,
    query_input_format: str,
    agent_input_format: str,
) -> datasets.IrtDataset:
    queries = datasets.load_dataset_hf(dataset_config.queries).with_format("torch")
    agents = datasets.load_dataset_hf(dataset_config.agents).with_format("torch")
    responses = datasets.load_dataset_hf(dataset_config.responses)
    return datasets.IrtDataset(
        responses, queries, agents, query_input_format, agent_input_format
    )


def make_dataloaders(config: RunConfig):
    agent_indexer = create_agent_indexer_from_dataset(config.data.train_set.agents)
    print("# Agents: ", agent_indexer.n_agents)
    print("# Agent Types: ", agent_indexer.n_agent_types)
    train_collator = collators.CaimiraCollator(agent_indexer, is_training=True)
    val_collator = collators.CaimiraCollator(agent_indexer, is_training=False)

    train_dataset = load_dataset(
        config.data.train_set,
        config.data.question_input_format,
        config.data.agent_input_format,
    )

    if config.trainer.sampler == "weighted":
        # TODO: Implement this
        raise NotImplementedError("Weighted sampler not implemented")
        logger.info("Using Weighted Sampler.")

        # Gather indices from train_dataset that correspond to subject types
        ds_indices = {
            i
            for i, e in enumerate(train_dataset)
            if "human" in agent_indexer.get_agent_type(e["agent_id"])
        }
        w_humans = 3.0
        w_ai = 1.0
        weights = [
            w_humans if i in ds_indices else w_ai for i in range(len(train_dataset))
        ]
        sampler = torch_data.WeightedRandomSampler(weights, len(weights))

    train_loader = torch_data.DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        collate_fn=train_collator,
    )
    val_loaders = {}
    for name, val_set in config.data.val_sets.items():
        val_ds = load_dataset(
            val_set, config.data.question_input_format, config.data.agent_input_format
        )
        val_loaders[name] = torch_data.DataLoader(
            val_ds,
            batch_size=config.trainer.batch_size,
            shuffle=False,
            collate_fn=val_collator,
        )
    labels_ctr = Counter(e["ruling"] for e in train_dataset)
    random_chance = max(labels_ctr.values()) / len(train_dataset)
    logger.info(f"Random chance Accuracy: {random_chance * 100:.1f}%")

    return train_loader, val_loaders


def main(args: argparse.Namespace) -> None:
    config: RunConfig = config_utils.load_config_from_namespace(args, RunConfig)

    exp_name = config.run_name or make_run_name(config)
    logger.info("Experiment Name: " + exp_name)

    train_loader, val_loaders_dict = make_dataloaders(config)
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
    logger.info(f"Saved model to {model_ckpt_path}")


def add_arguments(
    parser: Optional[argparse.ArgumentParser] = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()

    return config_utils.populate_parser_with_config_args(RunConfig, parser)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=RichHelpFormatter, description="Train IRT model."
    )
    parser = add_arguments(parser)
    args = parser.parse_args()
    logger.info(args)
    main(args)


# %%
