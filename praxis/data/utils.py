"""Utility functions for data loading and dataset management."""

import os
from typing import Any, Dict, List, Optional

from praxis.data.config import (
    DATASET_COLLECTIONS,
    DATASETS,
    DEFAULT_WEIGHT,
    DIR_WEIGHT,
    TOOLS_WEIGHT,
    resolve_task_type,
)
from praxis.data.datamodule import PraxisDataModule
from praxis.data.datasets import (
    GitHistoryDataset,
    HuggingfaceDataset,
    KBDataset,
    MultiDirectoryDataset,
    SyntheticPrintDataset,
    SyntheticToolCallingDataset,
)
from praxis.data.datasets.network_retry import (
    hf_offline,
    is_skippable_load_error,
    reset_hub_session,
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
    seq_curriculum: str = "fixed",
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

    # An unrecognized curriculum must fail here rather than silently degrade to
    # the fixed roll: experiment yaml values are applied with setattr and so
    # bypass the CLI's choice validation entirely.
    from praxis.data import SEQ_CURRICULUM_REGISTRY

    if seq_curriculum not in SEQ_CURRICULUM_REGISTRY:
        raise ValueError(
            f"Unknown seq_curriculum={seq_curriculum!r}. "
            f"Valid choices: {sorted(SEQ_CURRICULUM_REGISTRY)}"
        )
    # `probe` is armed by SequenceProbeCallback (it needs the trainer to reach
    # the validation data manager); `fixed` needs no state at all.

    train_data = []
    skipped_datasets = []  # (path, reason) for the web notification bell
    config = get_dataset_configs(train_datasets, validation_datasets, rl_type)

    # Start from a clean HTTP client. A prior fork (DataLoader workers, a
    # checkpoint-resume fork) can leave huggingface_hub's shared httpx client
    # closed-but-referenced, which makes every streaming load fail with
    # "client has been closed" until the process restarts. Resetting once here,
    # before any dataset loads, guarantees the first attempt gets a live client.
    if not hf_offline():
        reset_hub_session()

    if hf_offline():
        print(
            "[DATA] OFFLINE mode: hub datasets resolve from the local cache "
            "(streaming disabled); uncached ones are skipped."
        )
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
        if dataset_type == "huggingface":
            # The sampler latches offline mode only when the hub itself is
            # unreachable; a dataset-specific failure (or an uncached one
            # while offline) raises and is skipped rather than fatal.
            try:
                train_data.append(get_dataset(dataset_type, tokenizer, seed, c, *args))
            except Exception as e:
                if not is_skippable_load_error(e):
                    raise
                reason = "not in local cache" if hf_offline() else "load failed"
                print(
                    f"[DATA] skipping {c.get('path')} ({reason}): "
                    f"{type(e).__name__}"
                )
                skipped_datasets.append((c.get("path"), reason))
        else:
            train_data.append(get_dataset(dataset_type, tokenizer, seed, c, *args))

    if data_path:
        # One sampler per path: each directory is a distinct item in the
        # sampling weights (its own weight, novelty/loss EMA, and metrics
        # card) instead of every path pooling into one dataset whose share
        # was a single slot regardless of how many paths were given.
        paths = data_path if isinstance(data_path, list) else [data_path]
        seen_names = set()
        for p in paths:
            dataset = get_dataset("directory", tokenizer, seed, data_path=p, *args)
            # Basename collisions (e.g. two dirs both named "src") would merge
            # their metrics/EMA entries; qualify with the parent dir.
            if dataset.dataset_path in seen_names:
                parent = os.path.basename(os.path.dirname(p.rstrip("/\\")))
                dataset.dataset_path = f"{parent}/{dataset.dataset_path}"
            seen_names.add(dataset.dataset_path)
            train_data.append(dataset)
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
            try:
                validation_data.append(
                    get_dataset("huggingface", tokenizer, seed, c, *args)
                )
            except Exception as e:
                if not is_skippable_load_error(e):
                    raise
                reason = "not in local cache" if hf_offline() else "load failed"
                print(
                    f"[VALIDATION] skipping {c.get('path')} ({reason}): "
                    f"{type(e).__name__}"
                )
                skipped_datasets.append((c.get("path"), reason))

    # Surface skipped datasets on the web app's notification bell. One summary
    # event (not one per dataset); the automatic offline latch that used to
    # raise this is gone now that failures are handled per-dataset.
    if skipped_datasets:
        names = ", ".join(p for p, _ in skipped_datasets)
        try:
            from praxis.interface.state.live_metrics import LiveMetrics

            LiveMetrics().add_event(
                f"{len(skipped_datasets)} dataset(s) skipped (couldn't load; "
                f"network down or not cached): {names}. Other datasets are "
                "training normally; restart when the hub returns for the full mixture.",
                level="warning",
            )
        except Exception:
            pass

    train_dataloader = PraxisDataModule(
        train_data,
        validation_data,
        tokenizer,
        hparams["batch_size"],
        hparams["block_size"],
        hparams["sequence_multiplier_tiers"],
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
        dataset.task_type = resolve_task_type(args[0])
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
            dataset.task_type = resolve_task_type(cfg)
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
        dataset.task_type = resolve_task_type(dataset_config)
        return dataset
    elif format == "kb":
        dataset_config = args[0] if args else {}
        dataset = KBDataset(tokenizer, seed, dataset_config)
        dataset.weight = dataset_config.get("weight", DEFAULT_WEIGHT)
        dataset.task_type = resolve_task_type(dataset_config)
        return dataset
    elif format == "git_history":
        dataset_config = args[0] if args else {}
        dataset = GitHistoryDataset(tokenizer, seed, dataset_config)
        dataset.weight = dataset_config.get("weight", DEFAULT_WEIGHT)
        dataset.task_type = resolve_task_type(dataset_config)
        return dataset
    elif format == "synthetic-print":
        dataset_config = args[0] if args else {}
        dataset = SyntheticPrintDataset(tokenizer, seed, dataset_config)
        dataset.weight = dataset_config.get("weight", DEFAULT_WEIGHT)
        dataset.task_type = resolve_task_type(dataset_config)
        return dataset


def add_collection(config, collection_name, target_key, skip_existing=False):
    """Add datasets from a collection to the config with their weights.

    Args:
        config: Configuration dictionary to update
        collection_name: Name of the collection to add
        target_key: Key in config to add datasets to (primary or validation)
        skip_existing: Drop datasets already present under ``target_key``.
            Collections OVERLAP (``hh-rlhf`` is in both ``focused`` and
            ``preference``), and this function appends unconditionally, so a
            second collection carrying the same dataset would list it twice and
            double its sampling weight. Used by the rl_type auto-include path,
            where the collection is added on the model's behalf and must not
            reweight a mix the user set deliberately. Off by default: an
            explicit ``train_datasets`` listing the same dataset via two
            collections is asking for the combined weight.

    Returns:
        Updated configuration dictionary
    """
    if collection_name in DATASET_COLLECTIONS:
        present = {e.get("_id") for e in config[target_key]} if skip_existing else set()
        for dataset_name, weight in DATASET_COLLECTIONS[collection_name].items():
            if dataset_name in present:
                continue
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


def _rl_uses_datasets(rl_type) -> bool:
    """Whether any ``rl_type`` entry is a dataset-RL method needing RL collections.

    rl_type may be a single name or a list. Weight-editing controllers (e.g.
    ``harmonic_weight``) are marked ``is_weight_controller``; they reward
    loss-delta from a training callback and use no RL datasets, so the data
    pipeline must ignore them - otherwise setting their ``rl_type`` silently mixes
    RL data into training. Profile keys (e.g. ``harmonic_weight_wave``) are
    resolved to their underlying policy first. Deferred import keeps data/
    independent of policies/ at module load.
    """
    try:
        from praxis.policies import needs_rl_datasets, normalize_rl_types

        return any(needs_rl_datasets(name) for name in normalize_rl_types(rl_type))
    except Exception:
        return bool(rl_type)  # fail open: behave as before if the registry is absent


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
    requested = list(train_datasets)
    for name in requested:
        config = add_collection(config, name, "primary")

    from praxis.policies import normalize_rl_types, rl_dataset_collections

    # Each rl_type entry declares the collections it is bound to, and they are
    # force-included here so `rl_type` alone is enough to configure a working
    # RL task. Without this the two keys had to agree by hand, and disagreeing
    # failed silently: a policy filtering on a task tag no dataset emits scores
    # nothing and reports nothing. See praxis.policies.rl_dataset_collections.
    #
    # `skip_existing` is what keeps this side-effect-free for a mix the user
    # already set: add_collection APPENDS, collections overlap at the dataset
    # level (`hh-rlhf` is in both `focused` and `preference`), and a duplicate
    # entry does not error - it silently doubles that dataset's sampling
    # weight. Deduping by collection name alone is not enough for the same
    # reason.
    rl_names = normalize_rl_types(rl_type)
    seen_collections = set(requested)
    forced: List[str] = []
    for name in rl_names:
        for collection in rl_dataset_collections(name):
            if collection in seen_collections:
                continue
            seen_collections.add(collection)
            forced.append(collection)
            before = len(config["primary"])
            config = add_collection(config, collection, "primary", skip_existing=True)
            if len(config["primary"]) == before:
                forced.pop()  # every dataset was already in the mix

    for name in validation_datasets:
        config = add_collection(config, name, "validation")

    if rl_names:
        rl_count = len([e for e in config["primary"] if "RL" in e.get("path", "")])
        note = f", auto-added {forced}" if forced else ""
        print(
            f"[RL] RL enabled with {rl_names}, "
            f"{rl_count} RL datasets in config{note}"
        )

    return config
