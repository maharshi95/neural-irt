import argparse
import json
from typing import Any, List

import yaml

from neural_irt.utils.merge_utils import deep_merge_dict


def load_config_dict(filepath: str) -> dict:
    """Load a config dictionary from a file. Supports YAML and JSON."""
    with open(filepath, "r") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {filepath}")


def load_config_dicts(*filepaths: str) -> dict:
    """
    Parse a list of config files and return a dictionary.

    Args:
        *filepaths (str): The list of filepaths to parse.

    Returns:
        dict: The parsed dictionary.

    """
    if not filepaths:
        raise ValueError("At least one path is required")
    config = {}
    for filepath in filepaths:
        new_config = load_config_dict(filepath)
        config = deep_merge_dict(config, new_config)
    return config


def load_config_from_filepaths(paths: List[str], cls: type | None = None) -> Any:
    """Load a config from a list of filepaths.

    If cls is provided, the config is returned as an instance of cls.
    Otherwise, the config is returned as a dictionary.
    """

    config_dict = load_config_dicts(*paths)

    if cls:
        return cls(**config_dict)
    return config_dict


def load_config_from_namespace(
    args: argparse.Namespace, cls: type | None = None
) -> Any:
    config_dict = {}

    if args.config_paths:
        config_dict = load_config_from_filepaths(args.config_paths)

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
            value = load_config_dicts(value)

        current = config_dict
        for key in prefix_keys:
            if key not in current or current[key] is None:
                current[key] = {}
            current = current[key]
        current[current_key] = value

    # Let Pydantic handle the type conversion
    return cls(**config_dict) if cls else config_dict
