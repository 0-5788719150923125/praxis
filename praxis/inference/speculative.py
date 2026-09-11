"""MTP speculative decoding, as a transformers decoding method.

This is registered through ``generate(custom_generate=...)``, so transformers
runs its usual input preparation and then calls this INSTEAD of ``_sample``,
handing over a fully prepared ``LogitsProcessorList`` and
``StoppingCriteriaList``. Nothing in here rebuilds either: the repetition
penalty the terminal relies on, the suppressed control ids the byte classifier must
not sample, temperature/top-k/top-p, the request deadline, the chat format's
stop strings and its EOS ids all arrive already assembled and are simply used.

That is the whole point of the shape. The three Praxis decode loops each used
to re-derive this apparatus by hand from ``generation_config``, and they drifted
- a knob honored on one path was silently ignored on another, and every new
sampling feature had to be re-implemented three times or quietly not work.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers.generation.utils import GenerateDecoderOnlyOutput

from praxis.inference.decoding import (
    first_halt,
    is_halted,
    pick_next,
    stream_end,
    stream_put,
)

__all__ = [
    "speculative_decoding",
    "spec_logits_and_hidden",
    "verify_prefixes_batched",
]


@torch.no_grad()
def spec_logits_and_hidden(model, generated, attention_mask=None):
    """Vocab logits + the hidden the MTP drafts from, for one prefix.

    Byte-latent: the shared classifier classifies the byte-level decoder hidden
    (``_compute_logits`` returns it as ``hidden_states``), so drafting rides
    that same byte space. Token models: classifier over the trunk's last hidden.

    ``attention_mask`` right-pads a batch of ragged prefixes; the byte-latent
    core is padding-invariant, so masked rows read identically to their
    unpadded form (the basis for lossless batched-prefix verification).

    That same invariance is what lets the row be padded up to a bucket when one
    is in force (:mod:`praxis.inference.bucketing`), so a turn produces a
    handful of shapes instead of one per step. Outputs are trimmed back here,
    so every caller keeps indexing by absolute position into a row exactly as
    long as ``generated`` was.
    """
    from praxis.inference.bucketing import pad_for_decode
    from praxis.modeling import PraxisModel

    generated, attention_mask, true_len = pad_for_decode(generated, attention_mask)

    base_out = PraxisModel.forward(
        model, input_ids=generated, attention_mask=attention_mask
    )
    if model.encoder:
        logits, _, hidden, _ = model._compute_logits(
            base_out, generated, skip_logits=False, attention_mask=attention_mask
        )
    else:
        hidden = base_out.last_hidden_state
        logits = model.classifier(hidden)
    return logits[:, :true_len], hidden[:, :true_len]


@torch.no_grad()
def verify_prefixes_batched(model, generated, candidates):
    """Lossless byte-latent verification via batched truncated prefixes.

    For each ``k`` in ``1..n`` build ``P_k = generated + candidates[:k]`` and
    read the model's prediction at ``P_k``'s LAST real position. That read is
    causal (nothing follows it) and the byte-latent core is padding-invariant,
    so it equals exactly what byte-by-byte greedy would predict after
    committing ``candidates[:k]``. All ``n`` prefixes ride one right-padded,
    mask-gated forward instead of ``n`` sequential ones.

    Returns ``pred_logits[k-1]`` = next-token logits following
    ``generated + candidates[:k]``, shape ``[n, vocab]``.
    """
    g = generated.size(1)
    n = candidates.size(1)
    full = g + n
    device = generated.device
    # Right-pad with 0 and gate it off with `mask` below. 0 is a real byte
    # under a pure-byte tokenizer, so the mask - not the value - is what
    # makes these positions inert.
    rows = generated.new_zeros((n, full))
    mask = generated.new_zeros((n, full))
    last_pos = torch.empty(n, dtype=torch.long, device=device)
    for k in range(1, n + 1):
        length = g + k
        rows[k - 1, :length] = torch.cat([generated[0], candidates[0, :k]])
        mask[k - 1, :length] = 1
        last_pos[k - 1] = length - 1
    logits, _ = spec_logits_and_hidden(model, rows, attention_mask=mask)
    return logits[torch.arange(n, device=device), last_pos]


@torch.no_grad()
def speculative_decoding(
    model,
    input_ids: torch.LongTensor,
    logits_processor=None,
    stopping_criteria=None,
    generation_config=None,
    tokenizer: Any = None,
    streamer: Any = None,
    **model_kwargs: Dict[str, Any],
):
    """MTP-based speculative decoding for faster inference.

    ONE model forward per step. Each step:
    1. Draft N candidates from the hidden states carried out of the previous
       step's forward (the MTP bank; a few tenths of a percent of a forward)
    2. Run ONE forward over ``generated + candidates``
    3. Read position ``gen_len-1+k`` for the true next byte after
       ``generated + candidates[:k]``, and accept up to the first divergence
    4. Carry that same forward's hidden states into the next step

    This works because every stage of the stack is causal under append, so position
    ``t`` of a long row equals running the prefix ending at ``t`` on its own: the
    space patcher is prefix-monotone, the local conv encoder/decoder are causal, and
    ``decoder_patch_ids_from_lengths`` drops patch 0 so no byte reads the open
    patch. Classifiers that pool the whole sequence to route (see
    ``BaseClassifier.causal_readout``) break that, and keep the truncated-prefix verifier
    :func:`verify_prefixes_batched`.

    A step always commits one byte past the block it verified - the correction that
    ended a run, or the bonus that followed a full one - so no forward has seen that
    position and its hidden comes from ``mtp.bridge_hidden``. That estimate steers
    only the NEXT step's drafts; every committed byte is confirmed against a real
    read, so a poor bridge costs accept rate and never correctness.

    Greedy is lossless (up to floating-point argmax ties, where greedy is itself
    ill-defined). With ``do_sample`` acceptance stays equality-based, so sampling
    remains approximate. Candidate 0 is exempt from re-verification only when drawn
    from a MEASURED hidden, where accepting it is distribution-exact.

    HALTING is delegated whole to the prepared criteria. A step commits several
    bytes at once, so a boundary - an EOS, a stop string, the positional cap - can
    complete in the MIDDLE of a run and the criteria only report that it completed;
    :func:`first_halt` recovers the position and everything drafted past it is
    dropped. A deadline is deliberately not positional: it ends the loop but keeps
    what was produced.

    ``tokenizer`` and ``streamer`` arrive because
    ``PraxisForCausalLM._extract_generation_mode_kwargs`` puts them back -
    transformers drops both for a callable decoding method.
    """
    max_new_tokens = getattr(generation_config, "max_new_tokens", None) or 100
    do_sample = bool(getattr(generation_config, "do_sample", False))
    return_dict = bool(getattr(generation_config, "return_dict_in_generate", False))

    def result(seq):
        stream_end(streamer)
        if return_dict:
            return GenerateDecoderOnlyOutput(sequences=seq)
        return seq

    def pick(raw_logits, context_ids):
        """Argmax/sample from ``raw_logits`` ([1, vocab]) with the prepared
        processors evaluated over ``context_ids`` - the prefix this token
        follows, so context-dependent processors (the repetition penalty) see
        what byte-by-byte decoding would have shown them. That is what keeps
        greedy-with-penalty lossless against byte-by-byte greedy-with-penalty:
        the penalty is a deterministic function of the committed prefix."""
        return pick_next(raw_logits, context_ids, logits_processor, do_sample)

    generated = input_ids
    # Byte-latent keeps its byte table on the encoder side, so
    # get_input_embeddings() is None there; use the model's byte embeds.
    embed_fn = model.embeds if model.encoder else model.get_input_embeddings()
    num_new = 0

    # A classifier whose logits at position t depend on bytes after t (SMEAR-style
    # sequence pooling) cannot have a whole candidate block read out of one
    # row, and falls back to the truncated-prefix verifier below - correct,
    # but a full re-encode per candidate. Default False so a classifier that has
    # not declared the property takes the safe path.
    single_row = bool(getattr(model.classifier, "causal_readout", False))

    # Hidden states aligned with `generated`, so the MTP has something to
    # draft from without a forward of its own. Rebuilt from each step's
    # verify forward. `first_exact` records whether its LAST position came
    # from a real forward or from the MTP bridge.
    h_row = None
    first_exact = False

    while num_new < max_new_tokens:
        # Where this step's commits start, so the halt scan below only inspects
        # bytes this step produced - see first_halt's start_index contract.
        step_start = generated.size(1)
        gen_len = generated.size(1)

        if h_row is None:
            # Only the first step (or the fallback path, which drops h_row
            # every step because its verifier returns no hidden states).
            main_logits, h_row = spec_logits_and_hidden(model, generated)
            last_logits = main_logits[:, -1, :]
            first_exact = True
        else:
            # The trunk already ran over these positions last step; only the
            # classifier is re-applied, which is a few percent of a forward.
            last_logits = model.classifier(h_row)[:, -1, :]

        # First candidate: the model's own next-byte pick when `h_row` ends
        # on a measured position, a draft when it ends on the bridge. The
        # verify below reads position gen_len-1 either way, so this is
        # confirmed like any other candidate rather than trusted.
        token_0 = pick(last_logits, generated)
        token_0_2d = token_0.unsqueeze(1)

        # Draft additional tokens with MTP. The width is the run length
        # acceptance actually delivers (mtp.draft_width), not the trained
        # depth: candidates past the first divergence are discarded, but
        # each one still costs a sequential draft here and one more column
        # in the verify row. Cutting the width cannot change what gets
        # committed, only how much is thrown away, so greedy stays lossless.
        draft_ids = model.mtp.draft_next_tokens(
            h_row[:, -1:, :], token_0_2d, embed_fn, model.classifier
        )

        # Combine: first pick + drafts -> [batch, 1+N]
        candidates = torch.cat([token_0_2d, draft_ids], dim=1)
        n_candidates = candidates.size(1)

        # Greedy target following prefix P_k = generated + candidates[:k]; the
        # penalty context is that same prefix.
        def prefix_ids(k):
            if k == 0:
                return generated
            return torch.cat([generated, candidates[:, :k]], dim=1)

        if single_row:
            # ONE causal forward over prefix + candidates. Position
            # gen_len-1+k carries the true next-byte logits after
            # `generated + candidates[:k]` AND the hidden at that prefix's
            # last byte, so this single row supplies both the verification
            # and the next step's drafting state.
            row = torch.cat([generated, candidates], dim=1)
            verify_logits, verify_hidden = spec_logits_and_hidden(model, row)

            def raw_at(k):
                return verify_logits[:, gen_len - 1 + k, :]

        elif model.encoder:
            # Non-causal classifier: byte-latent must read each prefix's OWN last
            # position, batched behind a mask (lossless, but n re-encodes).
            pred_logits = verify_prefixes_batched(model, generated, candidates)
            verify_hidden = None

            def raw_at(k):
                return last_logits if k == 0 else pred_logits[k - 1 : k]

        else:
            verify_input = torch.cat([generated, candidates], dim=1)
            verify_logits, _ = spec_logits_and_hidden(model, verify_input)
            verify_hidden = None

            def raw_at(k):
                return verify_logits[:, gen_len - 1 + k, :]

        def target_at(k):
            return pick(raw_at(k), prefix_ids(k))

        def carry(exact_len, token):
            """Next step's `h_row`: measured through `exact_len` positions,
            bridged for the one byte past them."""
            if verify_hidden is None:
                return None
            exact = verify_hidden[:, :exact_len, :]
            bridged = model.mtp.bridge_hidden(
                exact[:, -1:, :], token.unsqueeze(1), embed_fn
            )
            return torch.cat([exact, bridged], dim=1)

        # Under sampling, a candidate 0 drawn from a MEASURED hidden is
        # already a valid draw from the conditional the verify would
        # re-sample; a fresh multinomial matches only with probability
        # sum(p^2), so re-rolling adds no correctness and only drags the
        # width EMA down. Greedy verifies from 0 - the read is real and
        # confirms by construction - which is what keeps it lossless when
        # candidate 0 came from the bridge instead.
        skip_first = do_sample and first_exact
        accepted = 1 if skip_first else 0
        for i in range(accepted, n_candidates):
            v_token = target_at(i)
            if v_token.item() == candidates[:, i].item():
                accepted += 1
            else:
                # Divergence: keep accepted prefix + the true greedy token.
                parts = [generated]
                if accepted > 0:
                    parts.append(candidates[:, :accepted])
                parts.append(v_token.unsqueeze(1))
                generated = torch.cat(parts, dim=1)
                num_new += accepted + 1
                # The run ended here, so the width this step used was right
                # (or too wide) - feed the observed run back.
                model.mtp.note_accepted(accepted)
                h_row = carry(gen_len + accepted, v_token)
                first_exact = False
                break
        else:
            # All candidates accepted - also take a bonus token.
            generated = torch.cat([generated, candidates], dim=1)
            num_new += n_candidates
            # The window filled: the run was at least this long, so the EMA
            # is pulled UP and the next step probes wider.
            model.mtp.note_accepted(n_candidates)
            h_row = verify_hidden
            first_exact = verify_hidden is not None

            if num_new < max_new_tokens:
                bonus = target_at(n_candidates)
                generated = torch.cat([generated, bonus.unsqueeze(1)], dim=1)
                num_new += 1
                h_row = carry(gen_len + n_candidates, bonus)
                first_exact = False

        # One halt scan per step, over exactly the bytes this step committed.
        # This is where an EOS accepted mid-run, a stop string completed
        # mid-run, and the positional cap all land: the criteria say a halt
        # happened, first_halt says where, and the drafted tail past it is
        # dropped rather than emitted on the model's behalf.
        cut = first_halt(generated, stopping_criteria, step_start)
        if cut is not None:
            generated = generated[:, :cut]
            stream_put(streamer, generated[:, step_start:])
            return result(generated)

        stream_put(streamer, generated[:, step_start:])

        # Non-positional criteria - the request deadline - end the loop without
        # truncating: a step is one forward, which is the unit of time this
        # bounds, and committed bytes are kept.
        if is_halted(generated, stopping_criteria):
            break

    return result(generated)
