"""Model and optimizer compilation utilities for torch.compile."""

import torch


def try_compile_model(model, hparams):
    """
    Attempt to compile the model with torch.compile.
    Falls back to uncompiled model if compilation fails or is not supported.

    Args:
        model: The model to compile
        hparams: Hyperparameters object or dict with configuration

    Returns:
        Compiled model or original model if compilation fails
    """
    # Convert dict to object if needed for getattr
    if isinstance(hparams, dict):
        from types import SimpleNamespace

        hparams = SimpleNamespace(**hparams)

    # Check if torch.compile is available (PyTorch 2.0+)
    if not hasattr(torch, "compile"):
        print(
            "[Compile] torch.compile not available (requires PyTorch 2.0+), using uncompiled model"
        )
        return model

    # Skip compilation for CPU (usually not beneficial)
    device = getattr(hparams, "device", "cpu")
    if "cpu" in str(device).lower():
        print("[Compile] Skipping compilation on CPU (not beneficial)")
        return model

    # Skip compilation in dev mode (faster iteration)
    if getattr(hparams, "dev", False):
        print("[Compile] Skipping compilation in dev mode")
        return model

    # Skip compilation for certain problematic configurations
    # MonoForward decoder has issues with compilation due to dynamic behavior
    if hasattr(hparams, "decoder_type") and hparams.decoder_type == "mono_forward":
        print("[Compile] Skipping compilation for mono_forward decoder (incompatible)")
        return model

    try:
        # Set TensorFloat32 for better performance
        torch.set_float32_matmul_precision("high")

        print("[Compile] Attempting model compilation...")

        # First try with most compatible settings
        # Note: 'default' mode balances speed and memory usage
        # 'reduce-overhead' uses less memory but may be slower
        # 'max-autotune' uses most memory but potentially fastest
        try:
            compiled_model = torch.compile(
                model,
                mode="default",  # ~30% more memory, good speedup
                fullgraph=False,
                dynamic=True,
            )

            # Quick sanity check without actual forward pass
            # Check if the compilation created the wrapper
            if (
                hasattr(compiled_model, "_orig_mod")
                or hasattr(compiled_model, "_torchdynamo_orig_callable")
                or "OptimizedModule" in type(compiled_model).__name__
            ):
                print("[Compile] ✓ Model compilation wrapper created successfully")
                print(
                    "[Compile] Note: Actual compilation will happen on first forward pass"
                )
                print(
                    "[Compile] Memory usage will increase by ~20-40% due to graph storage"
                )
                return compiled_model
            else:
                print(
                    "[Compile] Compilation did not create expected wrapper, using uncompiled model"
                )
                return model

        except Exception as compile_error:
            # If even basic compilation fails, try with minimal settings
            print(
                f"[Compile] Standard compilation failed: {str(compile_error)[:100]}..."
            )
            print("[Compile] Trying minimal compilation mode...")

            try:
                compiled_model = torch.compile(
                    model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    disable=False,
                    dynamic=False,  # Disable dynamic shapes
                )

                if (
                    hasattr(compiled_model, "_orig_mod")
                    or hasattr(compiled_model, "_torchdynamo_orig_callable")
                    or "OptimizedModule" in type(compiled_model).__name__
                ):
                    print("[Compile] ✓ Model compiled with minimal settings")
                    return compiled_model
                else:
                    print(
                        "[Compile] Minimal compilation failed, using uncompiled model"
                    )
                    return model

            except Exception:
                print(
                    "[Compile] All compilation attempts failed, using uncompiled model"
                )
                return model

    except Exception as e:
        # Fallback to uncompiled model if compilation fails
        print(f"[Compile] Unexpected error during compilation: {e}")
        print("[Compile] Falling back to uncompiled model")
        return model


def try_compile_optimizer(optimizer, hparams):
    """
    Attempt to compile the optimizer with torch.compile.
    Falls back to uncompiled optimizer if compilation fails.

    Args:
        optimizer: The optimizer to compile
        hparams: Hyperparameters object or dict with configuration

    Returns:
        Compiled optimizer or original optimizer if compilation fails
    """
    # Convert dict to object if needed for getattr
    if isinstance(hparams, dict):
        from types import SimpleNamespace

        hparams = SimpleNamespace(**hparams)

    # Check if torch.compile is available
    if not hasattr(torch, "compile"):
        return optimizer

    # Skip compilation for CPU
    device = getattr(hparams, "device", "cpu")
    if "cpu" in str(device).lower():
        return optimizer

    # Skip compilation in dev mode
    if getattr(hparams, "dev", False):
        return optimizer

    # Check if optimizer step can be compiled
    # Some optimizers don't work well with compilation
    optimizer_name = optimizer.__class__.__name__.lower()

    # Expanded list of problematic optimizers
    # ScheduleFree uses datetime.now() which can't be traced by Dynamo
    # LBFGS, Rprop, ASGD have complex control flow
    problematic_optimizers = ["lbfgs", "rprop", "asgd", "schedulefree", "schedule_free"]

    # Check for wrapper optimizers that might contain problematic optimizers
    if hasattr(optimizer, "__class__"):
        # Check class name and any base classes
        class_chain = [optimizer.__class__.__name__.lower()]
        if hasattr(optimizer, "_optimizer"):
            # For wrapped optimizers
            class_chain.append(optimizer._optimizer.__class__.__name__.lower())

        for class_name in class_chain:
            if any(opt in class_name for opt in problematic_optimizers):
                print(
                    f"[Compile] Skipping compilation for {optimizer.__class__.__name__} optimizer (contains datetime or complex control flow)"
                )
                return optimizer

    try:
        print("[Compile] Attempting optimizer compilation...")

        # Compile the optimizer's step method
        original_step = optimizer.step

        # Wrap the step method with torch.compile
        compiled_step = torch.compile(
            original_step,
            mode="default",  # Use same mode as model
            fullgraph=False,
            dynamic=True,
        )

        # Replace the step method
        optimizer.step = compiled_step

        # Check if compilation wrapper was created
        if hasattr(optimizer.step, "_torchdynamo_orig_callable"):
            print("[Compile] ✓ Optimizer compilation wrapper created successfully")
            print("[Compile] Note: Optimizer compilation happens on first step")
            return optimizer
        else:
            # Restore original if compilation didn't work
            optimizer.step = original_step
            print(
                "[Compile] Optimizer compilation did not create wrapper, using uncompiled"
            )
            return optimizer

    except Exception as e:
        # Fallback to uncompiled optimizer
        print(f"[Compile] Optimizer compilation failed: {str(e)[:100]}...")
        print("[Compile] Using uncompiled optimizer")
        # Ensure we restore the original step method if it was modified
        if hasattr(optimizer, "step") and hasattr(
            optimizer.step, "_torchdynamo_orig_callable"
        ):
            try:
                optimizer.step = optimizer.step._torchdynamo_orig_callable
            except:
                pass
        return optimizer
