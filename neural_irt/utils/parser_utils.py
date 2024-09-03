# %%
import argparse
from inspect import isclass
from types import NoneType, UnionType
from typing import Any, List, Union, get_args, get_origin

from loguru import logger
from pydantic import BaseModel

from neural_irt.utils import config_utils


def get_nested_types_of_union(union_type: type) -> List[type]:
    """Get the nested types from a union type. Leaves the other compositional types unchanged.
    Example:
    >>> get_nested_types(Union[List[Dict[str, int]], List[int]])
    [List[Dict[str, int]], List[int]]
    >>> get_nested_types(List[Dict[str, int]])
    [List[Dict[str, int]]]
    >>> get_nested_types(Dict[str, int])
    [Dict[str, int]]
    """
    if get_origin(union_type) in [Union, UnionType]:
        return [f for f in get_args(union_type) if f is not NoneType]
    return [union_type]


def is_union_type(field_type: type) -> bool:
    return get_origin(field_type) in [Union, UnionType]


def is_pydantic_model_type(field_type: type) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, BaseModel)


# %%
def add_filepath_argument_for_pydantic_model(
    parser: argparse.ArgumentParser, prefix: str, cls: type
):
    """Add a filepath argument to the parser for Pydantic model."""
    cls_name = cls.__name__ if isclass(cls) else cls
    parser.add_argument(
        f"--{prefix}",
        type=str,
        default=None,
        dest=f".config_path_{prefix}",
        help=f"Path to yaml config file for {prefix}: {cls_name}",
    )


def add_argument_for_pydantic_field(
    parser: argparse.ArgumentParser,
    arg_name: str,
    field_type: type,
    model_default: Any = None,
):
    """Add an argument to the parser for Pydantic field that is not a Pydantic model."""
    if get_origin(field_type) in [Union, UnionType]:
        field_types = [t for t in get_args(field_type) if t is not NoneType]
        # If there is only one type, use that type, otherwise use str and let Pydantic handle the conversion
        field_type = field_types[0] if len(field_types) == 1 else str

    help_str = f"Type: {field_type.__name__}, Default: {model_default}"
    # Default in argparse is None, so that we can ignore the fields where user didn't explicitly set a value
    parser.add_argument(f"--{arg_name}", type=field_type, default=None, help=help_str)


def add_arguments_for_type(
    parser: argparse.ArgumentParser,
    arg_name: str,
    field_type: type,
    field_default: Any = None,
    required: bool = False,
) -> List[str]:
    logger.debug(f"Adding arguments for {field_type.__name__} at {arg_name}")

    if is_union_type(field_type):
        nested_types = get_nested_types_of_union(field_type)
        return add_arguments_for_type(
            parser, arg_name, nested_types[0], field_default, required
        )

    required_fields = []
    if get_origin(field_type) in [dict, list]:
        # TODO: Handle this more gracefully.
        add_argument_for_pydantic_field(parser, arg_name, str, field_default)
    elif is_pydantic_model_type(field_type):
        required_fields = add_arguments_for_pydantic_model(parser, arg_name, field_type)
    else:
        add_argument_for_pydantic_field(parser, arg_name, field_type, field_default)
        if required:
            required_fields = [arg_name]
    return required_fields


def add_arguments_for_pydantic_model(
    parser: argparse.ArgumentParser,
    prefix: str,
    cls: type,
    *,
    add_config_path_arg: bool = True,
) -> List[str]:
    # If prefix is not empty (i.e. nested field),
    # add an argument that takes a path to a yaml file for the nested field

    if prefix and add_config_path_arg:
        add_filepath_argument_for_pydantic_model(parser, prefix, cls)

    required_fields = []
    for field_name, field in cls.model_fields.items():
        full_name = f"{prefix}.{field_name}" if prefix else field_name
        fields = add_arguments_for_type(
            parser,
            full_name,
            field.annotation,
            field.default,
            field.is_required(),
        )
        required_fields.extend(fields)
    return required_fields


def add_at_least_one_argument(
    parser: argparse.ArgumentParser, required_fields: List[str]
):
    class AtLeastOneArgument(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not namespace.config_paths:
                missing_required = [
                    field
                    for field in required_fields
                    if getattr(namespace, field.replace(".", "_"), None) is None
                ]
                if missing_required:
                    parser.error(
                        f"The following required arguments are missing: {', '.join(missing_required)}"
                    )
            setattr(namespace, self.dest, values)

    parser.add_argument("--at-least-one", action=AtLeastOneArgument, nargs=0)


def populate_parser_with_config_args(
    parser: argparse.ArgumentParser,
    config_class: type,
    *,
    add_config_paths: bool = True,
    root_name: str = "",
) -> argparse.ArgumentParser:
    if add_config_paths:
        parser.add_argument(
            "--config-paths", nargs="+", type=str, help="Paths to YAML config files"
        )

    add_arguments_for_pydantic_model(
        parser, root_name, config_class, add_config_path_arg=add_config_paths
    )

    # add_at_least_one_argument(parser, required_fields)
    return parser


if __name__ == "__main__":
    # Example usage
    from neural_irt.scripts.test_configs import RunConfig

    parser = populate_parser_with_config_args(RunConfig)
    args = parser.parse_args()
    config = config_utils.load_config_from_namespace(args, RunConfig)

    print(config)
