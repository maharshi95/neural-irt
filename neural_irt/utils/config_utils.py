import argparse
import json
from typing import Any

import yaml
from pydantic import BaseModel

from neural_irt.utils.merge_utils import deep_merge_dict


def save_config(config: dict | BaseModel, filepath: str):
    """Save a config to a file. Supports YAML and JSON."""
    if isinstance(config, BaseModel):
        config = config.model_dump()
    with open(filepath, "w") as f:
        if filepath.endswith(".json"):
            json.dump(config, f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            yaml.dump(config, f)
        else:
            raise ValueError(f"Unsupported file extension: {filepath}")


def _load_config_from_single_file(filepath: str) -> dict:
    """Load a config dictionary from a file. Supports YAML and JSON."""
    with open(filepath, "r") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {filepath}")


def load_config(*paths: str, cls: type | None = None) -> Any:
    """Load a config from a list of filepaths.

    Configs from multiple files are merged sequentially,
    with later files overriding earlier ones for duplicate keys.

    If cls is provided, the config is returned as an instance of cls.
    Otherwise, the config is returned as a dictionary.
    """

    if not paths:
        raise ValueError("At least one path is required")
    config = {}
    for filepath in paths:
        new_config = _load_config_from_single_file(filepath)
        config = deep_merge_dict(config, new_config)

    return cls(**config) if cls else config


def load_config_from_namespace(
    args: argparse.Namespace, cls: type | None = None
) -> Any:
    config_dict = {}

    if args.config_paths:
        config_dict = load_config(*args.config_paths)

    filepath_args = set()
    user_provided_args = {}

    for arg, value in vars(args).items():
        if arg in ["config_paths", "at_least_one"] or value is None:
            continue
        if arg.startswith(".config_path_"):
            arg = arg.removeprefix(".config_path_")
            filepath_args.add(arg)
        user_provided_args[arg] = value

    for arg, value in sorted(user_provided_args.items()):
        *prefix_keys, current_key = arg.split(".")
        if arg in filepath_args:
            value = load_config(value)

        current = config_dict
        for key in prefix_keys:
            if key not in current or current[key] is None:
                current[key] = {}
            current = current[key]
        current[current_key] = value

    # Let Pydantic handle the type conversion
    return cls(**config_dict) if cls else config_dict
