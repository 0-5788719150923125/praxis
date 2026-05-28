"""Process/environment bootstrap and model assembly for a training run."""

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict


def configure_torch_precision():
    """Opt into low-precision matmul kernels when the system supports them."""
    import torch

    try:
        torch.set_float32_matmul_precision("medium")
        print("[INIT] Your system will train with low-precision kernels.")
    except Exception as e:
        print(e)
        print("[INIT] Your system does not support low-precision kernels.")


def register_praxis_models():
    """Register Praxis config/model classes with the transformers Auto APIs."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    )

    from praxis import PraxisConfig, PraxisForCausalLM, PraxisModel

    AutoConfig.register("praxis", PraxisConfig)
    AutoModel.register(PraxisConfig, PraxisModel)
    AutoModelForCausalLM.register(PraxisConfig, PraxisForCausalLM)
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["praxis"] = "PraxisForCausalLM"


def setup_environment(no_docs: bool = False):
    """Set up warnings, env vars, docs, and model registration for a run."""
    from praxis.trainers.factory import disable_warnings
    from praxis.utils import check_for_updates

    sys.dont_write_bytecode = True
    configure_torch_precision()
    check_for_updates()

    if not no_docs:
        try:
            from praxis.docs import regenerate_docs

            regenerate_docs()
        except Exception as e:
            print(f"[DOCS] Skipped auto-regeneration: {e}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    disable_warnings()
    logging.getLogger("pytorch").setLevel(logging.ERROR)

    ignored_warnings = [
        ".*Checkpoint directory.*exists and is not empty*",
        ".*JAX is multithreaded, so this will likely lead to a deadlock*",
        ".*Total length of `list` across ranks is zero.*",
    ]
    for pattern in ignored_warnings:
        warnings.filterwarnings("ignore", pattern)

    register_praxis_models()


@dataclass
class ModelBundle:
    """The instantiated model and the values derived alongside it."""

    model: Any
    hparams: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    disable_schedule: bool
    total_params: int
    num_params: str


def assemble_model(cfg, config) -> ModelBundle:
    """Resolve the optimizer profile, build hparams, instantiate the model."""
    from transformers import AutoModelForCausalLM

    from praxis.optimizers import get_optimizer_profile
    from praxis.utils import initialize_lazy_modules

    optimizer_config, disable_schedule_from_optimizer = get_optimizer_profile(
        cfg.optimizer, any([cfg.fixed_schedule, cfg.schedule_free])
    )
    disable_schedule = cfg.disable_schedule or disable_schedule_from_optimizer

    config.optimizer_config = optimizer_config
    config.optimizer_wrappers = {
        "trac": cfg.trac,
        "ortho": cfg.ortho,
        "lookahead": cfg.lookahead,
        "schedule_free": cfg.schedule_free,
    }

    # Spread config first; explicit params override any duplicates.
    hparams = {
        **config.to_dict(),
        "batch_size": cfg.batch_size,
        "target_batch_size": cfg.target_batch_size,
        "block_size": cfg.block_size,
        "oversample_chance": 0.1,  # double the block_size
        "supersample_chance": 0.01,  # quadruple the block_size
        "hypersample_chance": 0.001,  # octuple the block_size
        "device": cfg.device,
        "trainer_type": cfg.trainer_type,
        "no_compile": cfg.no_compile,
    }

    model = AutoModelForCausalLM.from_config(config)
    initialize_lazy_modules(model, cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    num_params = f"{total_params / 10**6:.2f}M"
    hparams["num_params"] = num_params

    return ModelBundle(
        model=model,
        hparams=hparams,
        optimizer_config=optimizer_config,
        disable_schedule=disable_schedule,
        total_params=total_params,
        num_params=num_params,
    )


def build_model_info(cfg, config, bundle, run) -> Dict[str, Any]:
    """Build the model_info dict shown by TerminalInterface and the trainer."""
    from praxis.environments import EnvironmentFeatures

    return {
        "optimizer_config": bundle.optimizer_config,
        "strategy": cfg.strategy,
        "rl_type": cfg.rl_type,
        "vocab_size": cfg.vocab_size,
        "depth": config.depth,
        "num_layers": config.num_layers,
        "hidden_size": config.hidden_size,
        "embed_size": config.embed_size,
        "dropout": cfg.dropout,
        "dev": EnvironmentFeatures.get_active_environment() == "dev",
        "seed": cfg.seed,
        "truncated_hash": run.truncated_hash,
        "total_params": bundle.total_params,
        "batch_size": cfg.batch_size,
        "target_batch_size": cfg.target_batch_size,
    }
