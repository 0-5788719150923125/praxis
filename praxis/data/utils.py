"""Utility functions for data loading and dataset management."""

import os
from typing import Any, Dict, List, Optional

from praxis.data.config import (
    DATASET_COLLECTIONS,
    DATASETS,
    DIR_WEIGHT,
    TOOLS_WEIGHT,
)
from praxis.data.datamodule import PraxisDataModule
from praxis.data.datasets import (
    HuggingfaceDataset,
    MultiDirectoryDataset,
    SyntheticToolCallingDataset,
)


def get_datamodules(
    seed: int,
    train_datasets: Optional[List[str]],
    validation_datasets: Optional[List[str]],
    tokenizer,
    hparams,
    data_path,
    rl_type: Optional[str] = None,
    run_dir: Optional[str] = None,
    data_metrics_log_interval: int = 50,
    enable_chat_validation: bool = True,
    strict_chat_validation: bool = False,
    weighting_mode: str = "novelty",
    *args,
):
    """Create and configure data modules for training and validation.

    Args:
        seed: Random seed for dataset shuffling
        train_datasets: Named dataset collections for training (defaults to ["base"])
        validation_datasets: Named dataset collections for validation (defaults to ["validation"])
        tokenizer: Tokenizer to use for text processing
        hparams: Hyperparameters dictionary
        data_path: Ad-hoc directory paths (from --data-path) to append
        rl_type: Type of reinforcement learning to use
        *args: Additional arguments

    Returns:
        PraxisDataModule configured with train and validation datasets
    """

    train_datasets = list(train_datasets) if train_datasets else ["base"]
    validation_datasets = (
        list(validation_datasets) if validation_datasets else ["validation"]
    )
    train_data = []
    config = get_dataset_configs(train_datasets, validation_datasets, rl_type)

    for c in config["primary"]:
        dataset_type = c.get("type", "huggingface")
        if dataset_type == "huggingface":
            print(
                "[DATA] "
                + str(
                    dict(
                        path=c["path"],
                        weight=c["weight"],
                        keys=c.get("keys", ["text"]),
                    )
                )
            )
        else:
            print(
                f"[DATA] {dict(id=c.get('_id'), type=dataset_type, weight=c['weight'])}"
            )
        train_data.append(get_dataset(dataset_type, tokenizer, seed, c, *args))

    if data_path:
        train_data.append(
            get_dataset("directory", tokenizer, seed, data_path=data_path, *args)
        )
    # Load any module-provided datasets
    try:
        from praxis.cli import integration_loader

        available_datasets = integration_loader.integration_registry.get("datasets", {})
        for dataset_name in available_datasets:
            print(f"[INTEGRATIONS] Checking dataset: {dataset_name}")
            dataset = get_dataset(dataset_name, tokenizer, seed)
            if dataset is not None:
                print(f"[INTEGRATIONS] Adding dataset: {dataset_name}")
                train_data.append(dataset)
            else:
                print(
                    f"[INTEGRATIONS] Skipping dataset: {dataset_name} (not available)"
                )
    except ImportError:
        pass  # Integration loader not available

    validation_data = []
    if len(config["validation"]) > 0:
        for c in config["validation"]:
            print("[VALIDATION] " + str(dict(path=c["path"], weight=c["weight"])))
            validation_data.append(
                get_dataset("huggingface", tokenizer, seed, c, *args)
            )

    train_dataloader = PraxisDataModule(
        train_data,
        validation_data,
        tokenizer,
        hparams["batch_size"],
        hparams["block_size"],
        hparams["oversample_chance"],
        hparams["supersample_chance"],
        hparams["hypersample_chance"],
        rl_type=rl_type,
        run_dir=run_dir,
        data_metrics_log_interval=data_metrics_log_interval,
        enable_chat_validation=enable_chat_validation,
        strict_chat_validation=strict_chat_validation,
        weighting_mode=weighting_mode,
    )

    return train_dataloader


def get_dataset(format, tokenizer, seed, *args, **kwargs):
    """Get a dataset instance based on format type.

    Args:
        format: Type of dataset (huggingface, directory, self, synthetic-tool-calling)
        tokenizer: Tokenizer to use
        seed: Random seed
        *args: Additional positional arguments
        **kwargs: Additional keyword arguments

    Returns:
        Dataset instance or None if not available
    """
    # Check if this is a module-provided dataset
    try:
        from praxis.cli import integration_loader

        dataset_provider = integration_loader.get_dataset(format)
        if dataset_provider:
            # Call the provider function with standard arguments
            dataset = dataset_provider(tokenizer, seed, *args, **kwargs)
            # Check if the provider returned a valid dataset
            if dataset is not None:
                # Set default weight if not already set
                if not hasattr(dataset, "weight"):
                    dataset.weight = 1.0
                return dataset
            else:
                # Provider returned None (e.g., integration not properly initialized)
                return None
    except ImportError:
        pass

    if format == "huggingface":
        dataset = HuggingfaceDataset(tokenizer, seed, *args)
        dataset.weight = args[0].get("weight", 1.0)
        return dataset
    elif format == "directory":
        if args and isinstance(args[0], dict):
            cfg = args[0]
            directories = cfg.get("path")
            name = cfg.get("name") or "custom-files"
            extra = (
                {"allowed_extensions": cfg["allowed_extensions"]}
                if "allowed_extensions" in cfg
                else {}
            )
            dataset = MultiDirectoryDataset(
                tokenizer, directories=directories, name=name, **extra
            )
            dataset.weight = cfg.get("weight", DIR_WEIGHT)
            return dataset
        directories = kwargs.get("data_path")
        first = directories[0] if isinstance(directories, list) else directories
        name = os.path.basename(first.rstrip("/\\")) if first else "custom-files"
        dataset = MultiDirectoryDataset(tokenizer, directories=directories, name=name)
        dataset.weight = DIR_WEIGHT
        return dataset
    elif format == "synthetic-tool-calling":
        dataset_config = args[0] if args else {}
        dataset = SyntheticToolCallingDataset(tokenizer, seed, dataset_config)
        dataset.weight = dataset_config.get("weight", TOOLS_WEIGHT)
        return dataset


def add_collection(config, collection_name, target_key):
    """Add datasets from a collection to the config with their weights.

    Args:
        config: Configuration dictionary to update
        collection_name: Name of the collection to add
        target_key: Key in config to add datasets to (primary or validation)

    Returns:
        Updated configuration dictionary
    """
    if collection_name in DATASET_COLLECTIONS:
        for dataset_name, weight in DATASET_COLLECTIONS[collection_name].items():
            entry = DATASETS.get(dataset_name)
            if entry is None:
                raise ValueError(
                    f"Collection '{collection_name}' references unknown dataset "
                    f"'{dataset_name}'"
                )
            # Don't clobber an entry's own `name` (HuggingFace BuilderConfig);
            # carry the registry key under a private field for internal use.
            dataset_config = entry.copy()
            dataset_config["_id"] = dataset_name
            dataset_config["weight"] = weight
            config[target_key].append(dataset_config)
    return config


def get_dataset_configs(
    train_datasets: List[str],
    validation_datasets: List[str],
    rl_type: Optional[str] = None,
):
    """Get dataset configurations based on selected collections.

    Args:
        train_datasets: Named dataset collections for training
        validation_datasets: Named dataset collections for validation
        rl_type: Type of reinforcement learning

    Returns:
        Dictionary with primary and validation dataset configurations
    """
    unknown = [
        name
        for name in list(train_datasets) + list(validation_datasets)
        if name not in DATASET_COLLECTIONS
    ]
    if unknown:
        available = ", ".join(sorted(DATASET_COLLECTIONS.keys()))
        raise ValueError(
            f"Unknown dataset collection(s): {unknown}. Available: {available}"
        )

    config = {"primary": [], "validation": []}
    for name in train_datasets:
        config = add_collection(config, name, "primary")

    if rl_type:
        rl_collection = "cot" if rl_type in ["cot", "cot-reinforce"] else "rl"
        config = add_collection(config, rl_collection, "primary")

    for name in validation_datasets:
        config = add_collection(config, name, "validation")

    if rl_type:
        rl_count = len([e for e in config["primary"] if "RL" in e.get("path", "")])
        print(
            f"[RL] RL enabled with algorithm '{rl_type}', {rl_count} RL datasets in config"
        )

    return config
