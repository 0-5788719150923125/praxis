"""Model and optimizer compilation utilities for torch.compile."""

import torch

from praxis.environments import EnvironmentFeatures

COMPILE_KWARGS = dict(
    mode="default",  # ~30% more memory, good speedup
    fullgraph=False,
    # STATIC SHAPES, deliberately. `dynamic=True` makes every size symbolic from
    # the first trace, and Inductor's range inference then fails on this model
    # with `ValueRangeError: Invalid ranges [0:-1]` out of
    # index_propagation.py - a loop extent whose upper bound resolved to 0.
    # `dynamic=None` (trace static, re-trace symbolic once a size actually
    # moves) fails identically the moment a size moves. Measured on
    # abstractinator-q; only full specialization gets through.
    #
    # Affordable now in a way it was not under -g: static patching (-n) makes
    # the patch count a function of seq_len rather than of content, so the
    # shape set is small and bounded rather than open.
    dynamic=False,
)

# Graphs Dynamo may compile per frame before giving up and running eager.
#
# The default 8 exists to catch runaway retracing. The variation here is
# bounded and intentional, and 8 is simply too small for it: the recurrent
# loop passes `current_depth` as a Python int, so Dynamo guards on its VALUE
# and mints one graph per depth (6), and the batch governor varies microbatch
# rows (16 and 64 observed), which multiplies it. 6 x 2 = 12 > 8, so the
# router's forward was being abandoned to eager on a model that had just
# compiled successfully.
RECOMPILE_LIMIT = 32


def _hp(hparams, key, default=None):
    """Read a hyperparameter whether ``hparams`` is a dict or an object."""
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)


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

    if _hp(hparams, "no_compile", False):
        print("[COMPILER] Skipping compilation (--no-compile)")
        return model

    # Check if running on CPU - torch.compile has limited CPU support
    device = _hp(hparams, "device", "cpu")
    if isinstance(device, str) and device.startswith("cpu"):
        print(
            "[COMPILER] Skipping compilation (CPU device - limited torch.compile support)"
        )
        return model

    try:
        print("[COMPILER] Generating optimized kernel...")
        torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
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
