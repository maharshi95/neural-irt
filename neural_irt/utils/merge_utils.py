from typing import Any


class ListUpdateStrategy:
    """Strategy for overriding lists in deep_merge_list."""

    APPEND = "append"
    PREPEND = "prepend"
    REPLACE = "replace"
    FULL_REPLACE = "full_replace"
    RECURSIVE_REPLACE = "recursive_replace"


def deep_merge(
    base: Any,
    override: Any,
    list_update_strategy: str = ListUpdateStrategy.FULL_REPLACE,
) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        return deep_merge_dict(base, override, list_update_strategy)
    elif isinstance(base, list) and isinstance(override, list):
        return deep_merge_list(base, override, list_update_strategy)
    return override


def deep_merge_dict(
    base: dict,
    override: dict,
    list_update_strategy: str = ListUpdateStrategy.FULL_REPLACE,
) -> dict:
    """
    Deep merges two dictionaries recursively.

    Args:
        base (dict): The base dictionary.
        override (dict): The dictionary to override the base dictionary.

    Returns:
        dict: The merged dictionary.

    """
    result = base.copy()
    for key, value in override.items():
        if key in result:
            result[key] = deep_merge(result[key], value, list_update_strategy)
        else:
            result[key] = value
    return result


def deep_merge_list(
    base: list, override: list, strategy: str = ListUpdateStrategy.FULL_REPLACE
):
    if strategy == ListUpdateStrategy.FULL_REPLACE:
        return override
    elif strategy == ListUpdateStrategy.APPEND:
        return base + override
    elif strategy == ListUpdateStrategy.PREPEND:
        return override + base
    elif strategy == ListUpdateStrategy.REPLACE:
        return override + base[len(override) :]
    elif strategy == ListUpdateStrategy.RECURSIVE_REPLACE:
        new_list = []
        for base_item, override_item in zip(base, override):
            new_list.append(deep_merge(base_item, override_item, strategy))
        new_list.extend(base[len(override) :])
        return new_list
    else:
        raise ValueError(f"Unknown list update strategy: {strategy}")
