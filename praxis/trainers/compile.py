"""Model and optimizer compilation utilities for torch.compile."""

import torch


def _check_module_compilability(module):
    """
    Recursively check if any submodule is marked as non-compilable.

    Returns:
        tuple: (can_compile, module_path, module_type)
    """
    for name, submodule in module.named_modules():
        # Check for can_compile class variable on the submodule's class
        submodule_class = submodule.__class__
        if hasattr(submodule_class, "can_compile") and not submodule_class.can_compile:
            return False, name, submodule_class.__name__
    return True, None, None


def try_compile(obj, hparams):
    """
    Attempt to compile a PyTorch model or optimizer with torch.compile.
    Falls back to uncompiled object if compilation fails or is not supported.

    Args:
        obj: The model or optimizer to compile
        hparams: Hyperparameters object or dict with configuration

    Returns:
        Compiled object or original object if compilation fails
    """
    try:
        print("[COMPILER] Generating optimized kernel...")
        return torch.compile(
            obj,
            mode="default",  # ~30% more memory, good speedup
            fullgraph=False,
            dynamic=True,
        )
    except Exception as e:
        print(f"[COMPILER]\n")
        print(e)
        return obj
