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

    try:
        print("[COMPILER] Genererating optimized kernel...")
        return torch.compile(
            model,
            mode="default",  # ~30% more memory, good speedup
            fullgraph=False,
            dynamic=True,
        )
    except Exception as e:
        print(f"[COMPILER]\n")
        print(e)
    finally:
        return model

    # # TEMPORARY: Completely disable all model compilation to avoid FX tracing errors
    # print("[COMPILER] Model compilation is temporarily disabled")
    # return model

    # # Convert dict to object if needed for getattr
    # if isinstance(hparams, dict):
    #     from types import SimpleNamespace

    #     hparams = SimpleNamespace(**hparams)

    # # Check if torch.compile is available (PyTorch 2.0+)
    # if not hasattr(torch, "compile"):
    #     print(
    #         "[COMPILER] torch.compile not available (requires PyTorch 2.0+), using uncompiled model"
    #     )
    #     return model

    # # Skip compilation for CPU (usually not beneficial)
    # device = getattr(hparams, "device", "cpu")
    # if "cpu" in str(device).lower():
    #     print("[COMPILER] Skipping compilation on CPU (not beneficial)")
    #     return model

    # # Skip compilation if feature flag is set (faster iteration)
    # from praxis.environments import EnvironmentFeatures

    # if EnvironmentFeatures.is_enabled("skip_compilation"):
    #     print("[COMPILER] Skipping compilation (skip_compilation feature enabled)")
    #     return model

    # # Check if any module in the model is marked as non-compilable
    # # IMPORTANT: If ANY module cannot be compiled, we must skip ALL compilation
    # # to avoid FX tracing errors with partially compilable models
    # can_compile, module_path, module_type = _check_module_compilability(model)
    # if not can_compile:
    #     print(
    #         f"[COMPILER] Skipping ALL compilation: {module_type} at '{module_path}' marked as non-compilable"
    #     )
    #     print(
    #         "[COMPILER] Note: When any module is non-compilable, the entire model must remain uncompiled"
    #     )
    #     return model

    # # Skip compilation for certain problematic configurations
    # # MonoForward decoder has issues with compilation due to dynamic behavior
    # # if hasattr(hparams, "decoder_type") and hparams.decoder_type == "mono_forward":
    # #     print("[COMPILER] Skipping compilation for mono_forward decoder (incompatible)")
    # #     return model

    # print("[COMPILER] All modules are compilable, proceeding with compilation attempt")

    # try:
    #     # Set TensorFloat32 for better performance
    #     torch.set_float32_matmul_precision("high")

    #     print("[COMPILER] Attempting model compilation...")

    #     # First try with most compatible settings
    #     # Note: 'default' mode balances speed and memory usage
    #     # 'reduce-overhead' uses less memory but may be slower
    #     # 'max-autotune' uses most memory but potentially fastest
    #     try:
    #         compiled_model = torch.compile(
    #             model,
    #             mode="default",  # ~30% more memory, good speedup
    #             fullgraph=False,
    #             dynamic=True,
    #         )

    #         # Quick sanity check without actual forward pass
    #         # Check if the compilation created the wrapper
    #         if (
    #             hasattr(compiled_model, "_orig_mod")
    #             or hasattr(compiled_model, "_torchdynamo_orig_callable")
    #             or "OptimizedModule" in type(compiled_model).__name__
    #         ):
    #             print("[COMPILER] ✓ Model compilation wrapper created successfully")
    #             print(
    #                 "[COMPILER] Note: Actual compilation will happen on first forward pass"
    #             )
    #             print(
    #                 "[COMPILER] Memory usage will increase by ~20-40% due to graph storage"
    #             )
    #             return compiled_model
    #         else:
    #             print(
    #                 "[COMPILER] Compilation did not create expected wrapper, using uncompiled model"
    #             )
    #             return model

    #     except Exception as compile_error:
    #         # If even basic compilation fails, try with minimal settings
    #         print(
    #             f"[COMPILER] Standard compilation failed: {str(compile_error)[:100]}..."
    #         )
    #         print("[COMPILER] Trying minimal compilation mode...")

    #         try:
    #             compiled_model = torch.compile(
    #                 model,
    #                 mode="reduce-overhead",
    #                 fullgraph=False,
    #                 disable=False,
    #                 dynamic=True,  # Disable dynamic shapes
    #             )

    #             if (
    #                 hasattr(compiled_model, "_orig_mod")
    #                 or hasattr(compiled_model, "_torchdynamo_orig_callable")
    #                 or "OptimizedModule" in type(compiled_model).__name__
    #             ):
    #                 print("[COMPILER] ✓ Model compiled with minimal settings")
    #                 return compiled_model
    #             else:
    #                 print(
    #                     "[COMPILER] Minimal compilation failed, using uncompiled model"
    #                 )
    #                 return model

    #         except Exception:
    #             print(
    #                 "[COMPILER] All compilation attempts failed, using uncompiled model"
    #             )
    #             return model

    # except Exception as e:
    #     # Fallback to uncompiled model if compilation fails
    #     print(f"[COMPILER] Unexpected error during compilation: {e}")
    #     print("[COMPILER] Falling back to uncompiled model")
    #     return model


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
    # TEMPORARY: Completely disable all optimizer compilation to avoid FX tracing errors
    print("[COMPILER] Optimizer compilation is temporarily disabled")
    return optimizer

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

    # Skip compilation if feature flag is set
    from praxis.environments import EnvironmentFeatures

    if EnvironmentFeatures.is_enabled("skip_compilation"):
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
                    f"[COMPILER] Skipping compilation for {optimizer.__class__.__name__} optimizer (contains datetime or complex control flow)"
                )
                return optimizer

    try:
        print("[COMPILER] Attempting optimizer compilation...")

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
            print("[COMPILER] ✓ Optimizer compilation wrapper created successfully")
            print("[COMPILER] Note: Optimizer compilation happens on first step")
            return optimizer
        else:
            # Restore original if compilation didn't work
            optimizer.step = original_step
            print(
                "[COMPILER] Optimizer compilation did not create wrapper, using uncompiled"
            )
            return optimizer

    except Exception as e:
        # Fallback to uncompiled optimizer
        print(f"[COMPILER] Optimizer compilation failed: {str(e)[:100]}...")
        print("[COMPILER] Using uncompiled optimizer")
        # Ensure we restore the original step method if it was modified
        if hasattr(optimizer, "step") and hasattr(
            optimizer.step, "_torchdynamo_orig_callable"
        ):
            try:
                optimizer.step = optimizer.step._torchdynamo_orig_callable
            except:
                pass
        return optimizer
