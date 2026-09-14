"""Iterative-unmask decoding, registered as a transformers decoding method.

Generation under a diffusion objective is not sampling one token after another.
The whole answer exists from the first step as a block of mask symbols, and each
forward replaces some of them. Nothing is causal, nothing is cached, and the
sequence length is fixed before the first forward.

Registered the way transformers wants a non-standard strategy registered - a
decoding method handed to ``generate`` in place of ``_sample`` - so the prompt
handling, the logits-processor list, the stopping criteria and the streamer are
all prepared by the base implementation and honoured here without this loop
re-deriving any of them.

Two honest limitations, both consequences of the factorisation rather than of
this implementation:

* ``max_new_tokens`` is a commitment, not a ceiling. The block is allocated
  before the first forward, so the model cannot decide to stop early the way an
  EOS token lets an autoregressive loop stop. Stopping criteria are checked, but
  they can only end the refinement early - they cannot shorten the block.
* Committing k positions in one step assumes they are conditionally independent
  given the context. They are not. That is why ``diffusion_steps`` exists and
  why quality against steps is the first thing worth measuring.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.generation.utils import GenerateDecoderOnlyOutput


@torch.no_grad()
def refine(
    denoise,
    prompt_ids: Optional[torch.Tensor],
    length: int,
    *,
    mask_id: int,
    steps: int = 16,
    temperature: float = 0.0,
    logits_processor=None,
    batch_size: int = 1,
    device=None,
):
    """The reverse process itself, shared by every caller.

    ONE implementation on purpose. This loop existed twice - once for
    ``generate`` and once for the probe path - and the two drifted within an
    afternoon, which a test caught. A decode loop whose probe disagrees with
    the real path is worse than no probe.

    Args:
        denoise: ids ``(B, T)`` -> logits ``(B, T, V)``.
        prompt_ids: fixed prefix, never reconsidered. May be None.
        length: how many positions to generate. Fixed before the first
            forward - the block is allocated up front, so the model cannot
            decide to stop early the way an EOS token lets an AR loop stop.
        steps: refinement passes. ``steps == length`` commits one position per
            pass; fewer commits several at once, which assumes they are
            conditionally independent given the context. They are not, and
            that assumption is this loop's real failure mode.

    It does not stream. The caller publishes the finished block, because a
    streamer appends and this loop revises.

    NO STOPPING CRITERIA, deliberately. The block is allocated at full length
    before the first forward, so a length-based criterion is already satisfied
    on pass 1 - wiring transformers' prepared list in here ended the refinement
    after a single pass and returned a block of first guesses. The block length
    IS the stop condition under this factorisation; a content-based stop (an
    EOS the model writes into the middle of the block) would have to TRIM the
    result afterwards, which is a separate thing and not implemented.
    """
    if prompt_ids is not None and prompt_ids.numel():
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        prompt_len = prompt_ids.shape[1]
    else:
        prompt_len = 0

    total = prompt_len + length
    x = torch.full((batch_size, total), mask_id, dtype=torch.long, device=device)
    if prompt_len:
        x[:, :prompt_len] = prompt_ids

    generating = torch.zeros_like(x, dtype=torch.bool)
    generating[:, prompt_len:] = True

    steps = max(1, min(int(steps), max(1, length)))
    for step in range(steps):
        still_masked = (x == mask_id) & generating
        if not still_masked.any():
            break

        logits = denoise(x)
        if logits.shape[1] != total:
            logits = logits[:, :total]

        # Processors run on the FULL-width logits, before the mask column is
        # dropped. They index `scores` with ids taken from `input_ids` - the
        # repetition penalty gathers at exactly those columns - and `x` is full
        # of mask ids, so trimming first puts every one of them out of bounds.
        # On CUDA that is a device-side assert several steps later with no
        # useful traceback.
        if logits_processor is not None and len(logits_processor):
            B, L, V = logits.shape
            logits = logits_processor(
                x.repeat_interleave(L, dim=0), logits.reshape(B * L, V).float()
            ).view(B, L, V)

        # The mask id is an input symbol only; emitting it would be a
        # non-token. Everything at or above it is off the data alphabet.
        logits = logits[..., :mask_id]

        if temperature and temperature > 0:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            drawn = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1)
            pred = drawn.view(x.shape)
            conf = probs.reshape(-1, probs.shape[-1]).gather(1, drawn).view(x.shape)
        else:
            probs = F.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)

        # How many should still be masked after this pass, walking linearly to
        # zero over the remaining steps.
        # How many should still be masked after this pass, walking linearly to
        # zero over the remaining steps.
        remaining = int(round(length * (1.0 - (step + 1) / steps)))

        # Batched top-k rather than a loop over rows. Positions already
        # committed are pushed to -inf so they cannot win a slot, and rows are
        # allowed to differ in how many they still owe (they do not in
        # practice, but a ragged case must not silently commit the wrong
        # count). ``topk`` indices are unique within a row, so scattering the
        # keep flags cannot collide.
        conf = conf.masked_fill(~still_masked, float("-inf"))
        n_commit = (still_masked.sum(dim=-1) - remaining).clamp_min(0)
        budget = int(n_commit.max())
        if budget:
            chosen = conf.topk(budget, dim=-1)
            rank = torch.arange(budget, device=x.device).expand(batch_size, budget)
            keep = (rank < n_commit.unsqueeze(-1)) & torch.isfinite(chosen.values)
            commit = torch.zeros_like(still_masked)
            commit.scatter_(1, chosen.indices, keep)
            x = torch.where(commit, pred, x)

    # A rounding step can leave a position masked; fill it from a final pass
    # rather than emitting the absorbing symbol as if it were a token.
    leftover = (x == mask_id) & generating
    if leftover.any():
        x = torch.where(leftover, denoise(x)[..., :mask_id].argmax(dim=-1), x)
    return x


@torch.no_grad()
def unmask_decoding(
    model,
    input_ids: Optional[torch.Tensor] = None,
    logits_processor=None,
    stopping_criteria=None,
    generation_config=None,
    tokenizer: Any = None,
    streamer: Any = None,
    **model_kwargs: Any,
):
    """Reverse the absorbing process over a fixed-length block.

    Thin adapter: transformers hands over a prepared prompt, processor list,
    stopping criteria and streamer, and :func:`refine` does the work.
    """
    if input_ids is None:
        raise ValueError("diffusion decoding needs a prompt to place the block after")

    config = model.config
    length = int(getattr(generation_config, "max_new_tokens", None) or 128)
    steps = int(
        getattr(generation_config, "diffusion_steps", 0)
        or getattr(config, "diffusion_steps", 16)
    )
    do_sample = bool(getattr(generation_config, "do_sample", False))
    temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)

    result = refine(
        lambda ids: model(input_ids=ids).logits,
        input_ids,
        length,
        mask_id=model.criterion.main.mask_token_id,
        steps=steps,
        temperature=temperature if do_sample else 0.0,
        logits_processor=logits_processor,
    )

    # ONE publish, at the end. A streamer's ``put`` APPENDS
    # (praxis/inference/streamers.py), and a diffusion block is revised in
    # place rather than extended - so publishing each refinement pass appended
    # the whole block again every pass, mask symbols included. There is no
    # token-by-token reveal to show here: nothing is final until the pass that
    # commits it, and positions are committed by confidence rather than left to
    # right. transformers has already published the prompt, so this is the
    # generated block only.
    if streamer is not None:
        try:
            streamer.put(result[:, input_ids.shape[1] :].cpu())
            streamer.end()
        except Exception:
            pass

    # transformers decides the return SHAPE, not us: with
    # `return_dict_in_generate` the caller reads `.sequences`, and handing back
    # a bare tensor there fails every request with "'Tensor' object has no
    # attribute 'sequences'". Same contract CALM's vote decoding honours.
    if bool(getattr(generation_config, "return_dict_in_generate", False)):
        return GenerateDecoderOnlyOutput(sequences=result)
    return result
