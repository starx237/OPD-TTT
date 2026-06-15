# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import os
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Literal, Type, TypeVar, Union, get_type_hints

import yaml


try:
    from hdfs_io import copy, exists, makedirs  # for internal use only
except ImportError:
    from ..utils.hdfs_io import copy, exists, makedirs

from ..utils import helper, logging


logger = logging.get_logger(__name__)

T = TypeVar("T")


def _string_to_bool(value: Union[bool, str]) -> bool:
    """Converts a string representation of truth to True (1) or False (0)."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _deep_update(source: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update the source dictionary with the overrides dictionary.
    This ensures nested dictionaries are merged rather than overwritten.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = _deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def _map_flat_to_nested(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility: Map flat config structure to nested structure.
    This allows legacy configs with flat structure (e.g., enable_fsdp_offload)
    to work with the new nested structure (e.g., accelerator.fsdp_config.offload).
    """
    if not config:
        return config

    # Map train.* flat fields to nested structure
    train = config.get("train", {})
    if train:
        accelerator = train.get("accelerator", {})
        fsdp_config = accelerator.get("fsdp_config", {})
        offload_config = accelerator.get("offload_config", {})
        gradient_checkpointing = train.get("gradient_checkpointing", {})
        checkpoint = train.get("checkpoint", {})
        wandb = train.get("wandb", {})
        profile = train.get("profile", {})
        optimizer = train.get("optimizer", {})

        # Map flat fields to nested structure
        # use_wandb -> wandb.enable
        if "use_wandb" in train:
            wandb["enable"] = train["use_wandb"]

        # wandb_project -> wandb.project
        if "wandb_project" in train:
            wandb["project"] = train["wandb_project"]

        # wandb_name -> wandb.name
        if "wandb_name" in train:
            wandb["name"] = train["wandb_name"]

        # enable_fsdp_offload -> accelerator.fsdp_config.offload
        if "enable_fsdp_offload" in train:
            fsdp_config["offload"] = train["enable_fsdp_offload"]

        # enable_activation_offload -> accelerator.offload_config.enable_activation
        if "enable_activation_offload" in train:
            offload_config["enable_activation"] = train["enable_activation_offload"]

        # activation_gpu_limit -> accelerator.offload_config.activation_gpu_limit
        if "activation_gpu_limit" in train:
            offload_config["activation_gpu_limit"] = train["activation_gpu_limit"]

        # enable_gradient_checkpointing -> gradient_checkpointing.enable
        if "enable_gradient_checkpointing" in train:
            gradient_checkpointing["enable"] = train["enable_gradient_checkpointing"]

        # enable_reentrant -> gradient_checkpointing.enable_reentrant
        if "enable_reentrant" in train:
            gradient_checkpointing["enable_reentrant"] = train["enable_reentrant"]

        # enable_full_shard -> accelerator.fsdp_config.fsdp_mode = "fsdp2"
        if "enable_full_shard" in train and train["enable_full_shard"]:
            fsdp_config["fsdp_mode"] = "fsdp2"

        # data_parallel_mode -> accelerator.fsdp_config.fsdp_mode
        if "data_parallel_mode" in train:
            fsdp_config["fsdp_mode"] = train["data_parallel_mode"]

        # ulysses_parallel_size -> accelerator.ulysses_size
        if "ulysses_parallel_size" in train:
            accelerator["ulysses_size"] = train["ulysses_parallel_size"]

        # enable_mixed_precision -> accelerator.fsdp_config.mixed_precision.enable
        if "enable_mixed_precision" in train:
            if "mixed_precision" not in fsdp_config:
                fsdp_config["mixed_precision"] = {}
            fsdp_config["mixed_precision"]["enable"] = train["enable_mixed_precision"]

        # enable_forward_prefetch -> accelerator.fsdp_config.forward_prefetch
        if "enable_forward_prefetch" in train:
            fsdp_config["forward_prefetch"] = train["enable_forward_prefetch"]

        # Checkpoint fields: output_dir -> checkpoint.output_dir
        if "output_dir" in train:
            checkpoint["output_dir"] = train["output_dir"]

        # ckpt_manager -> checkpoint.manager
        if "ckpt_manager" in train:
            checkpoint["manager"] = train["ckpt_manager"]

        # save_steps -> checkpoint.save_steps
        if "save_steps" in train:
            checkpoint["save_steps"] = train["save_steps"]

        # save_epochs -> checkpoint.save_epochs
        if "save_epochs" in train:
            checkpoint["save_epochs"] = train["save_epochs"]

        # save_hf_weights -> checkpoint.save_hf_weights
        if "save_hf_weights" in train:
            checkpoint["save_hf_weights"] = train["save_hf_weights"]

        # Profile fields
        if "profile" in train and isinstance(train["profile"], bool):
            profile["enable"] = train["profile"]

        # Optimizer flat parameters (lr, lr_min, etc.) are handled in TrainingArguments.__post_init__
        # No need to map them here as they're synced after instantiation

        # Update nested structures back to config
        # Use _deep_update correctly: source = existing, overrides = new
        # Only call _deep_update when both source and overrides are dicts
        if wandb:
            existing = train.get("wandb", {})
            train["wandb"] = _deep_update(existing if isinstance(existing, dict) else {}, wandb)
        if checkpoint:
            existing = train.get("checkpoint", {})
            train["checkpoint"] = _deep_update(existing if isinstance(existing, dict) else {}, checkpoint)
        if profile:
            existing = train.get("profile", {})
            train["profile"] = _deep_update(existing if isinstance(existing, dict) else {}, profile)
        # Skip optimizer mapping here - it's handled in TrainingArguments.__post_init__
        # because optimizer can be a string (legacy format) or dict (nested format)
        if fsdp_config:
            existing = accelerator.get("fsdp_config", {})
            accelerator["fsdp_config"] = _deep_update(existing if isinstance(existing, dict) else {}, fsdp_config)
        if offload_config:
            existing = accelerator.get("offload_config", {})
            accelerator["offload_config"] = _deep_update(existing if isinstance(existing, dict) else {}, offload_config)
        # Always update accelerator with ulysses_size
        existing_acc = train.get("accelerator", {})
        train["accelerator"] = _deep_update(existing_acc if isinstance(existing_acc, dict) else {}, accelerator)
        if gradient_checkpointing:
            existing = train.get("gradient_checkpointing", {})
            train["gradient_checkpointing"] = _deep_update(existing if isinstance(existing, dict) else {}, gradient_checkpointing)
        config["train"] = train

    return config


# --- Recursive Argument Generation ---
def _add_arguments_recursive(parser: argparse.ArgumentParser, cls: Type[Any], prefix: str = ""):
    """
    Recursively traverse the Dataclass fields and generate arguments in the format
    --prefix.field.subfield for argparse.
    """
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    for field_info in dataclasses.fields(cls):
        field_name = field_info.name
        arg_name = f"{prefix}.{field_name}" if prefix else field_name

        field_type = type_hints.get(field_name, field_info.type)

        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if type(None) in args:
                field_type = args[0]

        if is_dataclass(field_type):
            _add_arguments_recursive(parser, field_type, prefix=arg_name)
        else:
            kwargs = {}
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                kwargs["choices"] = [e.value for e in field_type]
                kwargs["type"] = type(list(field_type)[0].value)
            elif field_type is bool:
                kwargs["type"] = _string_to_bool
                kwargs["nargs"] = "?"
                kwargs["const"] = True
            # Handle List (Simple handling, no deep recursion for lists of objects)
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                kwargs["nargs"] = "+"
                list_item_type = field_type.__args__[0]
                if list_item_type is bool:
                    kwargs["type"] = _string_to_bool
                else:
                    kwargs["type"] = list_item_type
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
                kwargs["choices"] = list(field_type.__args__)
                kwargs["type"] = type(field_type.__args__[0])
            else:
                kwargs["type"] = field_type

            if field_info.metadata and "help" in field_info.metadata:
                kwargs["help"] = field_info.metadata["help"]

            kwargs["default"] = argparse.SUPPRESS

            parser.add_argument(f"--{arg_name}", **kwargs)


def _instantiate_recursive(cls: Type[T], config_dict: Dict[str, Any]) -> T:
    """
    Recursively convert a dictionary into Dataclass instances.
    This triggers __post_init__ validation at every level.
    """
    if not is_dataclass(cls):
        return config_dict

    try:
        type_hints = get_type_hints(cls)
    except Exception:
        type_hints = {}

    field_values = {}
    for field_info in dataclasses.fields(cls):
        field_name = field_info.name

        # If the key is not in the config dict, skip it.
        # The dataclass will use its defined default_factory or default value.
        if field_name not in config_dict:
            continue

        raw_value = config_dict[field_name]

        # Prefer resolved type hint
        field_type = type_hints.get(field_name, field_info.type)

        # Unwrap Optional[T]
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            args = field_type.__args__
            if type(None) in args:
                field_type = args[0]

        # If the field expects a Dataclass and we have a dict, recurse
        if is_dataclass(field_type) and isinstance(raw_value, dict):
            field_values[field_name] = _instantiate_recursive(field_type, raw_value)
        else:
            field_values[field_name] = raw_value

    return cls(**field_values)


# --- Main Entry Point ---
def parse_args(root_class: Type[T]) -> T:
    """
    Parses arguments from both a YAML configuration file and Command Line Arguments.
    CLI arguments override YAML configurations.

    Supports both positional config_file and --config flag for compatibility.
    --config takes precedence over positional config_file.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # 兼容性支持：同时支持位置参数和 --config 标志
    parser.add_argument("config_file", nargs="?", help="Path to YAML config file (positional)")
    parser.add_argument("--config", dest="config_flag", help="Path to YAML config file (flag alias)")
    _add_arguments_recursive(parser, root_class)
    args = parser.parse_args()

    # 兼容性处理：--config 优先于位置参数 config_file
    config_path = getattr(args, "config_flag", None) or getattr(args, "config_file", None)

    final_config = {}

    if (
        config_path
        and (config_path.endswith(".yaml") or config_path.endswith(".yml"))
    ):
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                final_config = yaml_config

    cli_config = {}
    for key, value in vars(args).items():
        # 跳过配置文件参数（位置参数和标志参数）
        if key in ("config_file", "config_flag"):
            continue

        keys = key.split(".")
        current_level = cli_config
        for _i, k in enumerate(keys[:-1]):
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]
        current_level[keys[-1]] = value

    final_config = _deep_update(final_config, cli_config)

    # Backward compatibility: Map flat config structure to nested structure
    final_config = _map_flat_to_nested(final_config)

    return _instantiate_recursive(root_class, final_config)


def save_args(args: T, output_path: str) -> None:
    """
    Saves arguments to a yaml file.

    Args:
        args (dataclass): The arguments object.
        output_path (str): The destination path (supports HDFS if configured).
    """
    if output_path.startswith("hdfs://"):
        local_dir = helper.get_cache_dir()
        remote_dir = output_path
    else:
        logger.warning_once("Recommend to use hdfs path or hdfs_fuse path as the output path.")
        local_dir = output_path
        remote_dir = None

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "veomni_cli.yaml")

    # Save as YAML
    with open(local_path, "w") as f:
        f.write(yaml.safe_dump(asdict(args), default_flow_style=False))

    if remote_dir is not None:
        if not exists(remote_dir):
            makedirs(remote_dir)

        remote_path = os.path.join(remote_dir, "veomni_cli.yaml")
        copy(local_path, helper.convert_hdfs_fuse_path(remote_path))
