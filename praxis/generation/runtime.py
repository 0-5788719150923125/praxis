"""Inference wiring shared between the trainer and the API server."""

import torch


def bos_prompt(tokenizer):
    """Build the enable-prompt for the Mono-Forward live-inference hook.

    The hook seeds its streaming buffer from ``random_char_seed``, so this
    value is only the on/off gate: a non-``None`` return enables the hook.
    Returns a ``[1, 1]`` BOS id tensor, the plain ``bos_token`` string as a
    fallback, or ``None`` when the tokenizer has neither.
    """
    if tokenizer is None:
        return None
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return torch.tensor([[int(bos_id)]], dtype=torch.long)
    bos_token = getattr(tokenizer, "bos_token", None)
    return bos_token if isinstance(bos_token, str) and bos_token else None


def swap_inference_generator(trainer, tokenizer, api_server):
    """Route the API server through :class:`MonoForwardGenerator` for MF runs.

    The backprop flow builds a standard ``Generator`` before the trainer is
    known; that wraps ``model.generate()`` on the driver's CPU copy - wrong
    for Mono-Forward, whose trained weights live on Ray actors. For any
    non-MF trainer this early-returns and the backprop path is untouched.
    """
    if api_server is None:
        return
    try:
        from praxis.trainers.mono_forward import MonoForwardTrainer
    except ImportError:
        # Ray isn't installed; the factory would have raised earlier if the
        # user asked for --trainer-type mono_forward, so they didn't.
        return
    if not isinstance(trainer, MonoForwardTrainer):
        return

    from praxis.generation import MonoForwardGenerator
    from praxis.web.app import app

    mf_generator = MonoForwardGenerator(trainer=trainer, tokenizer=tokenizer)
    api_server.generator = mf_generator
    # Generation routes read app.config["generator"] separately from the
    # api_server reference updated above, so it needs an explicit assignment.
    app.config["generator"] = mf_generator
    print("[MF] Routed API server through MonoForwardGenerator")
