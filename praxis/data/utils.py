"""Utility functions for data loading and dataset management."""

from typing import Any, Dict, List, Optional

from praxis.data.config import (
    DATASET_COLLECTIONS,
    DIR_WEIGHT,
    HUGGINGFACE_DATASETS,
    SRC_WEIGHT,
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
    dev: bool,
    pile: bool,
    phi: bool,
    source: bool,
    tokenizer,
    hparams,
    data_path,
    rl_type: Optional[str] = None,
    run_dir: Optional[str] = None,
    data_metrics_log_interval: int = 50,
    enable_chat_validation: bool = True,
    strict_chat_validation: bool = False,
    *args,
):
    """Create and configure data modules for training and validation.

    Args:
        seed: Random seed for dataset shuffling
        dev: Use development dataset configuration
        pile: Use Pile dataset collection
        phi: Use Phi dataset collection and enable tool calling
        source: Include source code from current directory
        tokenizer: Tokenizer to use for text processing
        hparams: Hyperparameters dictionary
        data_path: Path to additional data directories
        rl_type: Type of reinforcement learning to use
        *args: Additional arguments

    Returns:
        PraxisDataModule configured with train and validation datasets
    """

    train_data = []
    config = get_dataset_configs(dev, pile, phi, rl_type)
    from praxis.environments import EnvironmentFeatures

    for c in config["primary"]:
        # load configs for huggingface datasets
        print(
            "[DATA] "
            + str(
                dict(path=c["path"], weight=c["weight"], keys=c.get("keys", ["text"]))
            )
        )
        train_data.append(get_dataset("huggingface", tokenizer, seed, c, *args))

    # Add synthetic tool-calling dataset if phi is enabled
    if phi:
        train_data.append(get_dataset("synthetic-tool-calling", tokenizer, seed))
        print("[CMD] Initialized SyntheticToolCallingDataset.")

    if not pile:
        if source:
            # load configs for training on praxis source code
            train_data.append(
                get_dataset(
                    "self",
                    tokenizer,
                    seed,
                    *args,
                )
            )
        # load configs for local file datasets
        if data_path:
            train_data.append(
                get_dataset(
                    "directory",
                    tokenizer,
                    seed,
                    data_path=data_path,
                    *args,
                )
            )
        # Load any module-provided datasets
        try:
            from praxis.cli import integration_loader

            available_datasets = integration_loader.integration_registry.get(
                "datasets", {}
            )
            # Process all available integration datasets
            # The integrations themselves will check if they're properly initialized
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
        dataset = MultiDirectoryDataset(
            tokenizer, directories=kwargs.get("data_path"), name="custom-files"
        )
        dataset.weight = DIR_WEIGHT
        return dataset
    elif format == "self":
        dataset = MultiDirectoryDataset(
            tokenizer,
            directories="./",
            name="src",
            allowed_extensions=[
                ".bib",
                ".cfg",
                ".css",
                ".gd",
                ".godot",
                ".html",
                ".ini",
                ".js",
                ".md",
                ".mjs",
                ".py",
                ".sh",
                ".tex",
                ".toml",
                ".ts",
                ".tscn",
                ".txt",
                ".yaml",
                ".yml",
                "LICENSE",
                "launch",
            ],
        )
        dataset.weight = SRC_WEIGHT
        return dataset
    elif format == "synthetic-tool-calling":
        dataset = SyntheticToolCallingDataset(tokenizer, seed, {})
        dataset.weight = TOOLS_WEIGHT
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
            dataset_config = HUGGINGFACE_DATASETS.get(dataset_name).copy()
            dataset_config["weight"] = weight
            config[target_key].append(dataset_config)
    return config


def get_dataset_configs(
    dev: bool, pile: bool, phi: bool, rl_type: Optional[str] = None
):
    """Get dataset configurations based on flags.

    Args:
        dev: Use development configuration
        pile: Use Pile dataset collection
        phi: Use Phi dataset collection
        rl_type: Type of reinforcement learning

    Returns:
        Dictionary with primary and validation dataset configurations
    """
    config = {"primary": [], "validation": []}
    if pile:
        config = add_collection(config, "pile", "primary")
        config = add_collection(config, "validation", "validation")
    else:
        config = add_collection(config, "base", "primary")
        if phi:
            config = add_collection(config, "phi", "primary")
        # Check for minimal_data feature flag
        from praxis.environments import EnvironmentFeatures

        if EnvironmentFeatures.is_enabled("minimal_data"):
            config["primary"] = []
            config = add_collection(config, "dev", "primary")
            # Add RL datasets even in dev mode if RL is enabled
            if rl_type:
                if rl_type in ["cot", "cot-reinforce"]:
                    config = add_collection(config, "cot", "primary")
                else:
                    config = add_collection(config, "rl", "primary")
        else:
            if rl_type:
                # Use different dataset collections based on RL type
                if rl_type in ["cot", "cot-reinforce"]:
                    config = add_collection(config, "cot", "primary")
                else:
                    config = add_collection(config, "rl", "primary")
            config = add_collection(config, "validation", "validation")

    # Debug: print RL status
    if rl_type:
        print(
            f"[RL] RL enabled with algorithm '{rl_type}', {len([e for e in config['primary'] if 'RL' in e.get('path', '')])} RL datasets in config"
        )

    return config
