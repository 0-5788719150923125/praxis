"""Model and optimizer compilation utilities for torch.compile."""

import torch

from praxis.environments import EnvironmentFeatures

COMPILE_KWARGS = dict(
    mode="default",  # ~30% more memory, good speedup
    fullgraph=False,
    dynamic=True,
)


def try_compile_model(model, hparams):
    """
    Attempt to compile a PyTorch model with torch.compile.
    Falls back to uncompiled model if compilation fails or is not supported.

    Args:
        model: The nn.Module to compile
        hparams: Hyperparameters object or dict with configuration

    Returns:
        Compiled model or original model if compilation fails
    """
    if EnvironmentFeatures.is_enabled("skip_compilation"):
        print("[COMPILER] Skipping compilation (skip_compilation feature enabled)")
        return model

    try:
        print("[COMPILER] Generating optimized kernel...")
        return torch.compile(model, **COMPILE_KWARGS)
    except Exception as e:
        print(f"[COMPILER]\n")
        print(e)
        return model


def try_compile_optimizer(optimizer):
    """
    Compile an optimizer's step function with torch.compile.

    torch.compile cannot be applied to optimizer objects directly — it requires
    a callable function. The correct pattern (per PyTorch docs) is to compile
    a function that calls optimizer.step().

    For wrapper optimizers (ScheduleFreeWrapper, Lookahead, OrthoGrad, TRAC),
    this walks the .optimizer chain to find the innermost optimizer and compiles
    its step method, since the wrapper's step() delegates to it.

    Args:
        optimizer: The optimizer (possibly wrapped) to compile

    Returns:
        The same optimizer object, with its step method compiled in-place
    """
    if EnvironmentFeatures.is_enabled("skip_compilation"):
        print("[COMPILER] Skipping compilation (skip_compilation feature enabled)")
        return optimizer

    # Walk the wrapper chain to find the innermost real optimizer
    target = optimizer
    wrapper_chain = []
    seen = set()
    while hasattr(target, "optimizer") and id(target) not in seen:
        seen.add(id(target))
        inner = target.optimizer
        if inner is target or not isinstance(inner, torch.optim.Optimizer):
            break
        wrapper_chain.append(type(target).__name__)
        target = inner

    target_name = type(target).__name__
    if wrapper_chain:
        chain_str = " -> ".join(wrapper_chain)
        print(f"[COMPILER] Unwrapping {chain_str} -> {target_name}")

    # Compile the step method of the innermost optimizer
    try:
        print(f"[COMPILER] Compiling {target_name}.step()...")
        target.step = torch.compile(target.step, **COMPILE_KWARGS)
        return optimizer
    except Exception as e:
        print(f"[COMPILER] Could not compile optimizer step: {e}")
        return optimizer


# Backward-compatible alias
def try_compile(obj, hparams):
    """
    Attempt to compile a PyTorch model or optimizer with torch.compile.

    Args:
        obj: The model or optimizer to compile
        hparams: Hyperparameters object or dict with configuration

    Returns:
        Compiled object or original object if compilation fails
    """
    if isinstance(obj, torch.optim.Optimizer):
        return try_compile_optimizer(obj)
    return try_compile_model(obj, hparams)
