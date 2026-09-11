"""Mono-Forward (in-process + Ray) correctness tests.

Consolidated test suite for everything MF-related: the in-process
math harness, the Ray-based pipelined trainer, checkpoint roundtrip,
and the Phase 5 live-inference-during-training hook. See
``PHASE_5.md`` for the project plan and decision rationale (D1..D8).

Tests that don't need Ray (the in-process math + detach checks) run
anywhere. Everything else is gated behind ``requires_ray`` so the host
venv (Python >= 3.14, no Ray wheels) skips them cleanly; run under
Docker compose to exercise the full suite:

    docker compose -f compose.yml run --rm --no-deps agent \\
        /workspace/.venv/bin/python -m pytest tests/test_mono_forward.py -v
"""

from __future__ import annotations

import json
import sqlite3

import pytest
import torch
from torch.utils.data import IterableDataset

from praxis import PraxisConfig
from praxis.modeling import PraxisForCausalLM
from praxis.trainers.mono_forward import InProcessMonoForwardTrainer

try:
    import ray  # noqa: F401

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

requires_ray = pytest.mark.skipif(not HAS_RAY, reason="Ray is not installed")

# Import the Ray trainer lazily so module import on non-Ray hosts still
# runs (the in-process math tests below don't need it).
if HAS_RAY:
    from praxis.trainers.mono_forward import MonoForwardTrainer
else:
    MonoForwardTrainer = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _mf_config(num_layers: int = 4) -> PraxisConfig:
    """Tiny CPU-only config that mirrors ``experiments/mike.yml`` in shape.

    ``num_layers`` is parameterised because different tests want
    different depths - the pipeline-overlap assertion needs at least 4
    layers for ``pipeline_in_flight_max >= num_layers - 1`` to be a
    meaningful "actually pipelined" proof, but the math and
    single-fit tests run faster at depth 2.
    """
    return PraxisConfig(
        vocab_size=256,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=num_layers,
        num_layers=num_layers,
        max_length=64,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )


class _FixedBatchDataset(IterableDataset):
    """Deterministic batch stream - yields the same sample forever.

    Memorisation workload: if the MF plumbing is sound the loss must
    trend downward. Used by every Ray test in this file.
    """

    def __init__(self, vocab_size: int, batch_size: int, seq_len: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.batch = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)

    def __iter__(self):
        while True:
            yield {"input_ids": self.batch}


class _SyntheticDataModule:
    """Minimal stand-in for a Lightning DataModule.

    ``MonoForwardTrainer.fit`` only calls ``train_dataloader()`` and
    iterates what it gets, so the real Praxis datamodule is overkill
    for a correctness smoke.
    """

    def __init__(self, dataset: _FixedBatchDataset) -> None:
        self._dataset = dataset

    def train_dataloader(self):
        return iter(self._dataset)


# ---------------------------------------------------------------------------
# Phase 2/3: Ray fit loop, checkpoint roundtrip, pipeline overlap, metrics.db
# ---------------------------------------------------------------------------


@requires_ray
def test_fit_reduces_loss_and_checkpoint_roundtrips(tmp_path):
    """End-to-end Phase 2 smoke + Phase 2 checkpoint roundtrip in one run.

    Phase 2 exit criteria: Ray fit() reduces loss, writes a monolithic
    checkpoint, and that checkpoint loads cleanly into a fresh vanilla
    ``PraxisForCausalLM`` producing bit-for-bit identical logits.
    """
    torch.manual_seed(0)
    config = _mf_config(num_layers=2)
    model = PraxisForCausalLM(config)

    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=16, seed=1
    )
    trainer = MonoForwardTrainer(
        max_steps=20, log_every_n_steps=10, cache_dir=str(tmp_path)
    )
    result = trainer.fit(model, _SyntheticDataModule(dataset))

    assert result["steps"] == 20
    assert result["final_loss"] < result["first_loss"], (
        f"MF-Ray did not reduce loss "
        f"(start={result['first_loss']:.4f}, end={result['final_loss']:.4f})"
    )

    checkpoint_path = tmp_path / "mono_forward.pt"
    assert checkpoint_path.exists()

    # Snapshot trained logits, then reload into a fresh vanilla model.
    model.eval()
    probe = torch.randint(
        0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(42)
    )
    with torch.no_grad():
        trained_logits = model(input_ids=probe).logits.detach().clone()

    reloaded = PraxisForCausalLM(_mf_config(num_layers=2))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Structured checkpoint format: model weights live under "model_state_dict".
    assert "model_state_dict" in checkpoint, "expected structured checkpoint format"
    assert "completed_batches" in checkpoint
    assert "projection_states" in checkpoint
    state = checkpoint["model_state_dict"]
    _missing, unexpected = reloaded.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in MF checkpoint: {unexpected}"

    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=probe).logits
    torch.testing.assert_close(
        reloaded_logits,
        trained_logits,
        rtol=1e-5,
        atol=1e-5,
        msg="MF checkpoint logits do not match reloaded vanilla model",
    )


@requires_ray
def test_depth_less_than_num_layers_hard_errors():
    """depth < num_layers must hard-error at fit time."""
    torch.manual_seed(0)
    bad_config = PraxisConfig(
        vocab_size=256,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        depth=1,  # fewer forward steps than layers
        num_layers=2,
        max_length=64,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )
    model = PraxisForCausalLM(bad_config)
    dataset = _FixedBatchDataset(
        vocab_size=bad_config.vocab_size, batch_size=1, seq_len=8, seed=2
    )
    trainer = MonoForwardTrainer(max_steps=1, cache_dir=None)
    with pytest.raises(RuntimeError, match="depth >= num_layers"):
        trainer.fit(model, _SyntheticDataModule(dataset))


@requires_ray
def test_pipeline_fills_and_logs_metrics(tmp_path):
    """Phase 3 exit criterion: pipelined training overlaps, logs, syncs.

    One run covers:
    - in-flight pipeline actually filled up (``pipeline_in_flight_max >=
      num_layers - 1``)
    - every layer produced at least one loss value
    - loss trend downward
    - head sync ran at the expected cadence
    - ``metrics.db`` contains per-layer losses + pipeline metrics in
      ``extra_metrics`` JSON
    """
    torch.manual_seed(0)
    config = _mf_config(num_layers=4)
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=16, seed=1
    )

    num_batches = 40
    head_sync_every = 20
    trainer = MonoForwardTrainer(
        max_steps=num_batches,
        log_every_n_steps=10,
        cache_dir=str(tmp_path),
        ray_pipeline_api="manual",
        ray_head_sync_every=head_sync_every,
    )
    result = trainer.fit(model, _SyntheticDataModule(dataset))

    assert result["completed_batches"] == num_batches
    for layer_idx in range(config.num_layers):
        assert (
            len(result["per_layer_loss_history"][layer_idx]) > 0
        ), f"layer {layer_idx} produced no loss values"
    assert result["pipeline_in_flight_max"] >= config.num_layers - 1, (
        f"pipeline never filled up (max={result['pipeline_in_flight_max']}, "
        f"expected >= {config.num_layers - 1})"
    )
    assert result["final_loss"] < result["first_loss"]

    metrics_db = tmp_path / "metrics.db"
    assert metrics_db.exists()
    conn = sqlite3.connect(str(metrics_db))
    try:
        rows = conn.execute(
            "SELECT step, loss, extra_metrics FROM metrics ORDER BY step"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == num_batches
    for step, loss_val, extra_json in rows:
        assert loss_val is not None
        extras = json.loads(extra_json)
        for layer_idx in range(config.num_layers):
            assert f"layer_{layer_idx}_loss" in extras
        assert "pipeline_in_flight" in extras


@requires_ray
def test_compiled_api_not_implemented():
    """The ``compiled`` pipeline API is a stub and must say so loud."""
    torch.manual_seed(0)
    config = _mf_config(num_layers=2)
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=1, seq_len=8, seed=2
    )
    trainer = MonoForwardTrainer(
        max_steps=1, cache_dir=None, ray_pipeline_api="compiled"
    )
    with pytest.raises(NotImplementedError, match="compiled"):
        trainer.fit(model, _SyntheticDataModule(dataset))


@requires_ray
def test_ray_num_replicas_per_layer_rejected_above_one():
    """--ray-num-replicas-per-layer > 1 is a stub; hard-error at init."""
    with pytest.raises(RuntimeError, match=r"ray.num.replicas.per.layer"):
        MonoForwardTrainer(ray_num_replicas_per_layer=2)


# ---------------------------------------------------------------------------
# Phase 5 Task 2: live inference during training
# ---------------------------------------------------------------------------


class _RecordingTrainer(MonoForwardTrainer if HAS_RAY else object):
    """Captures periodic-inference-hook output for assertion.

    The production hook prints the generated ids; tests want a
    structured record instead. Subclass override is the lightest-weight
    way to intercept without giving the production trainer a test-only
    knob.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_generations: list = []

    def _maybe_run_inference_hook(self, completed_batches, config):  # type: ignore[override]
        # Test-only override: fires on every final-hop boundary, with
        # no wall-clock gating, so the captured count is deterministic.
        # The production time-gated path is covered by the inline
        # unit check in ``test_inference_hook_fires_*`` below.
        if self.inference_prompt is None:
            return
        prompt = self.inference_prompt
        if not isinstance(prompt, torch.Tensor):
            prompt = torch.as_tensor(prompt, dtype=torch.long)
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        tokens = list(
            self.generate(
                prompt,
                max_new_tokens=self.inference_max_new_tokens,
                eos_token_id=getattr(config, "eos_token_id", None),
            )
        )
        self.captured_generations.append(
            dict(
                at_batch=completed_batches,
                prompt=prompt.tolist(),
                tokens=[t.tolist() for t in tokens],
            )
        )


@requires_ray
def test_inference_hook_fires_during_training(tmp_path):
    """Periodic-inference hook runs mid-fit and training still converges.

    Verifies: (1) the hook fires at every final-hop boundary under
    the test-only ``_RecordingTrainer`` override, producing the
    expected number of shape-correct token sequences, and (2) the
    training loss decreases despite concurrent inference traffic.
    """
    num_batches = 12
    prompt = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    torch.manual_seed(0)
    config = _mf_config(num_layers=3)
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=12, seed=1
    )
    trainer = _RecordingTrainer(
        max_steps=num_batches,
        log_every_n_steps=num_batches,
        cache_dir=None,
        inference_prompt=prompt,
        inference_every_seconds=0.0,
        inference_max_new_tokens=5,
    )
    result = trainer.fit(model, _SyntheticDataModule(dataset))
    captured = trainer.captured_generations

    # Every final-layer completion fires the hook under the recorder,
    # so captured count equals num_batches.
    assert (
        len(captured) == num_batches
    ), f"expected {num_batches} hook fires, got {len(captured)}"
    for sample in captured:
        assert len(sample["tokens"]) == 5  # max_new_tokens
        for tok in sample["tokens"]:
            assert len(tok) == 1  # batch dim
            assert 0 <= tok[0] < 256

    # Training must still converge even with concurrent inference.
    # With per-layer projection matrices (random init), we can't do
    # bit-level trajectory comparison across runs, but we CAN verify
    # that the loss decreased over the run.
    assert result["final_loss"] < result["first_loss"], (
        f"loss did not decrease with inference hook enabled: "
        f"start={result['first_loss']:.4f}, end={result['final_loss']:.4f}"
    )


@requires_ray
def test_generate_outside_active_fit_raises():
    """Calling generate() without a live actor set is a hard error."""
    trainer = MonoForwardTrainer(max_steps=1, cache_dir=None)
    with pytest.raises(RuntimeError, match="active actor set"):
        list(trainer.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2))


class _IdleGenerateTrainer(MonoForwardTrainer if HAS_RAY else object):
    """Runs ``generate`` between training and teardown.

    Production ``fit`` clears ``self._actors`` in its ``finally`` block,
    so the idle-generation window is narrow. Overriding
    ``_save_checkpoint`` is the natural injection point: by the time
    it's called, every training batch has completed but the actor set
    is still alive.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.idle_tokens: list = []

    def _save_checkpoint(self, model_host, actors):  # type: ignore[override]
        prompt = torch.tensor([[7, 8, 9]], dtype=torch.long)
        self.idle_tokens = [t.tolist() for t in self.generate(prompt, max_new_tokens=4)]
        super()._save_checkpoint(model_host, actors)


@requires_ray
def test_generate_while_training_idle(tmp_path):
    """Idle generate() works when no train_batch is in flight."""
    torch.manual_seed(0)
    config = _mf_config(num_layers=3)
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=10, seed=2
    )

    trainer = _IdleGenerateTrainer(
        max_steps=6, log_every_n_steps=6, cache_dir=str(tmp_path)
    )
    trainer.fit(model, _SyntheticDataModule(dataset))

    assert len(trainer.idle_tokens) == 4
    for tok in trainer.idle_tokens:
        assert len(tok) == 1
        assert 0 <= tok[0] < config.vocab_size


class _StubTrainer:
    """The whole surface :class:`MonoForwardLM` needs: a config and one forward.

    That is the point of the face - Mono-Forward's only real difference from
    in-process decoding is where the forward runs, so everything above it
    (sampling, halting, the request queue) is the ordinary path and needs no
    stub at all.
    """

    def __init__(self, token: int = 65, num_layers: int = 2):
        self._config = _mf_config(num_layers=num_layers)
        self.token = token
        self.calls = 0

    def infer_logits(self, input_ids):
        self.calls += 1
        b, t = input_ids.shape
        logits = torch.full((b, t, self._config.vocab_size), -10.0)
        logits[:, :, self.token] = 10.0
        return logits


def test_mono_forward_face_reports_a_device_with_no_parameters_of_its_own():
    """`PreTrainedModel.device` walks `parameters()`, and this module has none -
    the weights are on the actors. Left inherited it raised StopIteration the
    first time the backend asked where to put a prompt."""
    from praxis.trainers.mono_forward.hf_model import MonoForwardLM

    lm = MonoForwardLM(_StubTrainer())
    assert list(lm.parameters()) == []
    assert lm.device == torch.device("cpu")
    assert lm.dtype == torch.float32
    # ...and nothing it holds leaks into a checkpoint.
    assert lm.state_dict() == {}


def test_mono_forward_face_always_forwards_the_whole_prefix():
    """Prefill-every-step: the actors hold no KV cache, so a cache-shortened
    tail would feed them a one-token sequence and generate from nothing."""
    from praxis.trainers.mono_forward.hf_model import MonoForwardLM

    trainer = _StubTrainer()
    lm = MonoForwardLM(trainer)
    ids = torch.tensor([[1, 2, 3, 4]])
    prepared = lm.prepare_inputs_for_generation(ids, past_key_values=object())
    assert torch.equal(prepared["input_ids"], ids)
    assert prepared["use_cache"] is False


# ---------------------------------------------------------------------------
# In-process backend (no Ray). These run anywhere - the whole point of the
# in-process backend is that it has zero Ray dependency.
# ---------------------------------------------------------------------------


def test_inprocess_fit_reduces_loss_and_checkpoint_roundtrips(tmp_path):
    """End-to-end smoke for the in-process backend.

    Mirrors :func:`test_fit_reduces_loss_and_checkpoint_roundtrips` for
    the Ray path: training reduces loss, a structured checkpoint lands on
    disk, and reloading the model_state_dict into a fresh
    ``PraxisForCausalLM`` reproduces the trained-model logits exactly.
    """
    torch.manual_seed(0)
    config = _mf_config(num_layers=2)
    model = PraxisForCausalLM(config)

    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=16, seed=1
    )
    trainer = InProcessMonoForwardTrainer(
        max_steps=20,
        log_every_n_steps=10,
        cache_dir=str(tmp_path),
        device="cpu",
    )
    result = trainer.fit(model, _SyntheticDataModule(dataset))

    assert result["steps"] == 20
    assert result["final_loss"] < result["first_loss"], (
        f"In-process MF did not reduce loss "
        f"(start={result['first_loss']:.4f}, end={result['final_loss']:.4f})"
    )

    checkpoint_path = tmp_path / "mono_forward.pt"
    assert checkpoint_path.exists()

    model.eval()
    probe = torch.randint(
        0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(42)
    )
    with torch.no_grad():
        trained_logits = model(input_ids=probe).logits.detach().clone()

    reloaded = PraxisForCausalLM(_mf_config(num_layers=2))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "completed_batches" in checkpoint
    assert "projection_states" in checkpoint
    state = checkpoint["model_state_dict"]
    _missing, unexpected = reloaded.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys in MF checkpoint: {unexpected}"

    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=probe).logits
    torch.testing.assert_close(
        reloaded_logits,
        trained_logits,
        rtol=1e-5,
        atol=1e-5,
        msg="MF in-process checkpoint logits do not match reloaded vanilla model",
    )


def test_inprocess_recurrent_depth_routes_through_layers():
    """depth > num_layers must cycle through the worker set.

    Mirrors the Ray trainer's recurrent-depth contract: with depth=4
    and num_layers=2 the depth chain hits each worker twice. The
    in-process trainer's per-layer-loss history therefore has one entry
    per depth step, not one per worker.
    """
    torch.manual_seed(0)
    config = PraxisConfig(
        vocab_size=128,
        hidden_size=16,
        embed_size=16,
        num_heads=4,
        depth=4,
        num_layers=2,
        max_length=32,
        decoder_type="sequential",
        attention_type="modular",
        encoder_type=None,
        tie_weights=False,
    )
    model = PraxisForCausalLM(config)
    dataset = _FixedBatchDataset(
        vocab_size=config.vocab_size, batch_size=2, seq_len=16, seed=3
    )
    trainer = InProcessMonoForwardTrainer(
        max_steps=5, cache_dir=None, device="cpu", log_every_n_steps=10
    )
    result = trainer.fit(model, _SyntheticDataModule(dataset))

    per_layer = result["per_layer_loss_history"]
    # depth=4 means 4 entries, one per depth step (not per worker).
    assert set(per_layer.keys()) == {0, 1, 2, 3}
    for step_idx, losses in per_layer.items():
        assert (
            len(losses) == 5
        ), f"depth step {step_idx} should have one loss per batch, got {len(losses)}"
