"""Tests for the trainers module."""

import math

import pytest
import torch
from transformers import AutoTokenizer

from praxis import PraxisConfig, PraxisForCausalLM, registry
from praxis.optimization import get_optimizer, get_optimizer_profile
from praxis.schedulers import get_scheduler_func
from praxis.trainers import (
    BackpropagationTrainer,
    create_trainer_with_module,
    try_compile,
)


class TestBackpropagationTrainer:
    """Test cases for BackpropagationTrainer."""

    @pytest.fixture
    def setup_model(self):
        """Create a small model for testing."""
        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=100,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        return model, config

    @pytest.fixture
    def setup_tokenizer(self):
        """Create a simple tokenizer for testing."""

        # Create a mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
            sep_token_id = 3

            def encode(self, text, return_tensors=None):
                # Simple mock encoding
                if return_tensors == "pt":
                    return torch.tensor([[1, 2, 3, 4, 5]])
                return [1, 2, 3, 4, 5]

            def decode(self, ids, skip_special_tokens=False):
                return "test output"

        return MockTokenizer()

    def test_trainer_initialization(self, setup_model, setup_tokenizer):
        """Test basic BackpropagationTrainer initialization."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        # Setup optimizer and scheduler
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        # Create trainer
        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_level=False,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.tokenizer is tokenizer
        assert (
            trainer.outputs_are_aligned is False
        )  # byte_level=False means outputs_are_aligned=False

    def test_trainer_forward_with_kwargs(self, setup_model, setup_tokenizer):
        """Test that BackpropagationTrainer forward accepts keyword arguments."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_level=False,
        )

        batch_size = 2
        seq_len = 10

        # Test with keyword arguments (matching training_step usage)
        outputs = trainer.forward(
            input_ids=torch.randint(0, 100, (batch_size, seq_len)),
            labels=torch.randint(0, 100, (batch_size, seq_len - 1)),
        )
        assert outputs is not None
        assert hasattr(outputs, "loss")

    def test_trainer_forward_pass(self, setup_model, setup_tokenizer):
        """Test forward pass through BackpropagationTrainer."""
        model, config = setup_model
        tokenizer = setup_tokenizer

        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        scheduler_func = get_scheduler_func(optimizer_config)
        scheduler = scheduler_func(optimizer)

        trainer = BackpropagationTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            hparams={"batch_size": 4, "device": "cpu"},
            tokenizer=tokenizer,
            byte_level=False,
        )

        batch_size = 2
        seq_len = 10
        inputs = {
            "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
            "labels": torch.randint(0, 100, (batch_size, seq_len - 1)),
        }

        # Forward pass with dict unpacking
        outputs = trainer.forward(**inputs)
        assert outputs is not None
        assert hasattr(outputs, "loss")


class TestCompilationUtils:
    """Test compilation utilities."""

    def test_try_compile_with_model(self):
        """Test that try_compile works with models."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        hparams = {"device": "cpu"}

        compiled = try_compile(model, hparams)
        # Should return a model (original or compiled)
        assert compiled is not None

    def test_try_compile_with_optimizer(self):
        """Test that try_compile works with optimizers."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=10,
        )
        model = PraxisForCausalLM(config)
        optimizer_config, _ = get_optimizer_profile("AdamW")
        optimizer = get_optimizer(model, **optimizer_config)
        hparams = {"device": "cpu"}

        compiled = try_compile(optimizer, hparams)
        # Should return an optimizer (original or compiled)
        assert compiled is not None


class TestTrainerFactory:
    """Test the trainer factory function."""

    def test_create_trainer_with_module(self, tmpdir):
        """Test creating a trainer with module."""
        config = PraxisConfig(
            depth=1,
            hidden_size=32,
            embed_size=16,
            vocab_size=100,
        )
        model = PraxisForCausalLM(config)

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
            vocab_size = 100

        tokenizer = MockTokenizer()

        # Create a temporary checkpoint directory
        checkpoint_dir = str(tmpdir.mkdir("checkpoints"))

        trainer, trainer_module = create_trainer_with_module(
            trainer_type="backpropagation",
            model=model,
            tokenizer=tokenizer,
            hparams={
                "batch_size": 4,
                "device": "cpu",
                "learning_rate": 1e-3,
                "max_epochs": 1,
                "accumulate_grad_batches": 1,
                "gradient_clip_val": 1.0,
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_every_n_steps": 100,
            },
            experiment_name="test",
            run_name="test_run",
        )

        assert trainer is not None
        assert trainer_module is not None
        assert isinstance(trainer_module, BackpropagationTrainer)

    def test_trainer_registry(self):
        """Test that trainer registry contains expected trainers."""
        assert "backpropagation" in registry.namespace("trainers")
        assert "mono_forward" in registry.namespace("trainers")
        assert "mono_forward_ray" in registry.namespace("trainers")

        # Test that backpropagation trainer is directly accessible
        assert registry.lookup("trainers", "backpropagation") == BackpropagationTrainer

        # Both Mono-Forward profiles are lazy loaders so the package
        # (and Ray) only get imported when the profile is actually
        # selected.
        assert callable(registry.lookup("trainers", "mono_forward"))
        assert callable(registry.lookup("trainers", "mono_forward_ray"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestTrainingStepRunsEndToEnd:
    """The step itself, not just construction.

    Nothing here exercised `training_step` before, which is how a refactor
    that dropped a local binding shipped: `rewards` was still read further
    down the step, so EVERY step raised NameError while every existing test
    still passed.
    """

    @pytest.fixture
    def parts(self):
        import time

        from praxis.data.datasets.message_queue import MessageQueueManager
        from praxis.tokenizers import create_tokenizer

        tokenizer = create_tokenizer(
            vocab_size=1024, tokenizer_type="byte_level", chat_format="prose"
        )
        manager = MessageQueueManager(
            tokenizer=tokenizer, block_size=256, enable_chat_validation=False
        )
        conversation = [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ]
        for _ in range(12):
            manager.add_document({"messages": conversation, "metadata": {}})
        packed = manager.get_batch(batch_size=2)

        # Exactly the dict WeightedIterableDataset yields to the trainer.
        batch = {
            "input_ids": torch.stack(packed["batch"]),
            "metadata": packed["metadata"],
            "task_type_ids": torch.stack(packed["task_type_ids"]),
            "assistant_mask": torch.stack(packed["assistant_mask"]),
            "block_ids": torch.stack(packed["block_ids"]),
        }

        config = PraxisConfig(
            vocab_size=1024,
            byte_vocab_size=tokenizer.byte_alphabet_size,
            byte_offset=tokenizer.byte_offset,
            hidden_size=64,
            embed_size=32,
            num_heads=2,
            depth=2,
            num_layers=2,
            encoder_type="abstractinator_harmonic_gdn_vocab_bank",
            decoder_type="sequential",
            block_size=256,
            max_position_embeddings=1024,
            device_map="cpu",
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = PraxisForCausalLM(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        trainer = BackpropagationTrainer(
            model, optimizer, scheduler, {"batch_size": 2}, tokenizer, byte_level=True
        )
        # Lightning attributes the step touches but that no Trainer supplies here.
        trainer.trainer = type("_Stub", (), {"should_stop": False})()
        trainer.log_dict = lambda *a, **k: None
        trainer.log = lambda *a, **k: None
        trainer.last_train_step_time = time.monotonic() - 1.0
        return trainer, batch, tokenizer

    def test_training_step_completes(self, parts):
        trainer, batch, _ = parts
        loss = trainer.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_validation_step_completes(self, parts):
        trainer, batch, _ = parts
        trainer.validation_step(batch, 0)

    def test_step_runs_without_the_optional_channels(self, parts):
        """A bare tensor batch must still work: no block_ids, no masks."""
        trainer, batch, _ = parts
        loss = trainer.training_step(batch["input_ids"], 0)
        assert torch.isfinite(loss)

    def test_packed_batch_is_pure_bytes_with_real_block_ids(self, parts):
        """Guards the two halves of the pure-byte layout together."""
        _, batch, tokenizer = parts
        assert tokenizer.byte_alphabet_size == 256
        assert int(batch["input_ids"].max()) < 256
        assert batch["block_ids"].shape == batch["input_ids"].shape
        # More than one document per row, or packing is not being exercised.
        assert len(set(batch["block_ids"][0].tolist())) > 1


class TestRLCTProbeUnpacksTheBatch:
    """The RLCT callback re-uses the trainer's batch unpacking.

    `on_train_batch_end` swallows probe exceptions and prints them, so a break
    here is non-blocking and invisible to every other test - which is how an
    UnboundLocalError for `rewards` survived a whole training run. `_probe`
    does its unpacking OUTSIDE any try, so calling it directly makes that
    class of failure loud.
    """

    def test_probe_unpacks_without_unbound_names(self):
        from praxis.callbacks.lightning.rlct import RLCTLandscapeCallback

        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=256,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        pl_module = BackpropagationTrainer(
            model, optimizer, None, {"batch_size": 4}, None, byte_level=True
        )

        batch = {
            "input_ids": torch.randint(0, 256, (4, 16)),
            "task_type_ids": torch.zeros(4, 16, dtype=torch.uint8),
            "assistant_mask": torch.ones(4, 16, dtype=torch.uint8),
            "block_ids": torch.ones(4, 16, dtype=torch.long),
        }

        callback = RLCTLandscapeCallback(
            {"probe_seqs": 2, "probe_len": 8, "manifold_grid": 2, "field_grid": 2}
        )
        # Any NameError/UnboundLocalError in the unpacking propagates from here.
        callback._probe(pl_module, batch, 0, 0)

    def test_probe_forwards_block_ids(self):
        """block_ids must survive sub-batching, or the probe measures a model
        whose packed documents can read each other while the real step's cannot.
        """
        from praxis.callbacks.lightning import rlct

        captured = {}
        config = PraxisConfig(
            depth=2,
            hidden_size=64,
            embed_size=32,
            vocab_size=256,
            num_heads=2,
            num_queries=2,
            device_map="cpu",
        )
        model = PraxisForCausalLM(config)
        original = model.forward

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        model.forward = spy
        pl_module = BackpropagationTrainer(
            model,
            torch.optim.SGD(model.parameters(), lr=1e-4),
            None,
            {"batch_size": 4},
            None,
            byte_level=True,
        )
        batch = {
            "input_ids": torch.randint(0, 256, (4, 16)),
            "block_ids": torch.ones(4, 16, dtype=torch.long),
        }
        rlct.RLCTLandscapeCallback(
            {"probe_seqs": 2, "probe_len": 8, "manifold_grid": 2, "field_grid": 2}
        )._probe(pl_module, batch, 0, 0)

        assert captured.get("block_ids") is not None
        assert captured["block_ids"].shape == captured["input_ids"].shape


def test_byte_nll_bits_is_calibrated_against_chance():
    """The whole point of this metric is that its LEVEL means something.

    The series this replaced was the objective / ln(2), so under a composite
    loss it could sit anywhere (HALO runs opened above 16 on a scale whose
    maximum is 8). This one has to read 8.0 for a uniform 256-way predictor and
    0.0 for a certain one, or it cannot be compared to a scaling law.

    The trainer is not constructed here - `_compute_byte_nll_bits` only touches
    `self.outputs_are_aligned`, so a stub isolates the arithmetic from the
    Lightning module's dependencies.
    """
    import math
    import types

    import torch

    from praxis.trainers.backpropagation import BackpropagationTrainer

    stub = types.SimpleNamespace(
        outputs_are_aligned=True,
        _compute_byte_nll_bits=BackpropagationTrainer._compute_byte_nll_bits,
    )

    def run(logits, labels):
        out = types.SimpleNamespace(logits=logits)
        return stub._compute_byte_nll_bits(stub, out, labels)

    labels = torch.randint(0, 256, (2, 32))

    # Chance: uniform over 256 bytes is exactly 8 bits, by definition.
    uniform = torch.zeros(2, 32, 256)
    assert abs(float(run(uniform, labels)) - 8.0) < 1e-4

    # Certainty: all mass on the right byte is 0 bits.
    certain = torch.full((2, 32, 256), -1e4)
    certain.scatter_(-1, labels.unsqueeze(-1), 1e4)
    assert float(run(certain, labels)) < 1e-3

    # Padding must not be averaged in. Masking half the targets to the ignore
    # index has to leave a uniform predictor at 8.0, not pull it toward 0.
    masked = labels.clone()
    masked[:, ::2] = -100
    assert abs(float(run(uniform, masked)) - 8.0) < 1e-4


def test_byte_nll_bits_shifts_for_unaligned_encoders():
    """An unaligned encoder's last position has no target, and the caller has
    already shifted the labels - so the logits must lose their last step. If the
    two conventions ever drift apart, the metric silently scores position t
    against byte t+1 and the number stops being comparable to anything."""
    import types

    import torch

    from praxis.trainers.backpropagation import BackpropagationTrainer

    def run(aligned, logits, labels):
        stub = types.SimpleNamespace(outputs_are_aligned=aligned)
        out = types.SimpleNamespace(logits=logits)
        return BackpropagationTrainer._compute_byte_nll_bits(stub, out, labels)

    labels = torch.randint(0, 256, (2, 31))
    logits = torch.zeros(2, 32, 256)

    # Unaligned: 32 logits, 31 labels - the trim makes them meet.
    assert abs(float(run(False, logits, labels)) - 8.0) < 1e-4

    # Aligned with a mismatched shape returns None rather than raising: a
    # missing series is readable, an exception inside validation is not.
    assert run(True, logits, labels) is None


def test_bits_per_byte_is_reported_only_as_a_likelihood():
    """`val_bits_per_byte` used to be `val_loss / ln(2)` - a unit conversion of
    a series already charted, carrying no information, and inheriting whatever
    the training objective was. It is gone. The reported bits-per-byte is now
    `val_byte_nll_bits`, unweighted CE on the emitted logits.

    Pinned because the tempting "fix" is to redefine the old key in place, and
    a series that changes meaning under a stable name is worse than one that
    ends.
    """
    from praxis.metrics.training_metrics import TRAINING_METRIC_REGISTRY as REG
    from praxis.pillars.runs import METRIC_LABELS, METRIC_PRIORITY

    assert "val_bits_per_byte" not in REG
    assert "val_bits_per_byte" not in METRIC_LABELS
    assert "val_bits_per_byte" not in METRIC_PRIORITY
    # The calibrated series takes its place, at the front and on the chart.
    assert METRIC_PRIORITY[0] == "val_byte_nll_bits"
    assert REG["val_byte_nll_bits"]["chart"]["title"] == "Bits per Byte"
    # Codec fidelity keeps its own series and its own nats-to-bits helper.
    assert "val_codec_bpb" in REG


def test_nats_to_bits_helper_is_codec_only_and_correct():
    import math
    import types

    import torch

    from praxis.trainers.backpropagation import BackpropagationTrainer

    stub = types.SimpleNamespace()
    got = BackpropagationTrainer._compute_bits_per_byte(stub, torch.tensor(3.0))
    assert abs(float(got) - 3.0 / math.log(2)) < 1e-6


@pytest.mark.parametrize(
    "tag,over",
    [
        ("plain", {}),
        ("byte_latent_conv", {"encoder_type": "byte_latent_conv", "vocab_size": 1024}),
        (
            "abstractinator",
            {
                "encoder_type": "abstractinator_v1",
                "vocab_size": 1024,
                "codebook_size": 256,
            },
        ),
    ],
)
def test_byte_nll_bits_is_emitted_for_every_byte_level_family(tag, over):
    """Any byte-level tokenizer needs this metric, not just codec encoders.

    The gate is `self.byte_level`, which comes from the TOKENIZER
    (`cli/config.py`: `tokenizer_type == "byte_level"`), and the emission sits
    outside the codec branch - so a vanilla byte-latent run with no codec, and a
    plain model with no encoder at all, both get it. Only `val_codec_bpb` is
    encoder-gated.

    Pinned across families because this is now the front of METRIC_PRIORITY: a
    family that silently stops emitting it drops out of every run comparison.
    """
    import types

    import torch

    from praxis import PraxisConfig
    from praxis.modeling import PraxisForCausalLM
    from praxis.trainers.backpropagation import BackpropagationTrainer

    cfg = dict(
        vocab_size=256,
        hidden_size=64,
        embed_size=64,
        num_heads=2,
        depth=2,
        max_length=512,
        decoder_type="sequential",
        head_type="forward",
        tokenizer_type="byte_level",
        encoder_type=None,
    )
    cfg.update(over)
    torch.manual_seed(0)
    model = PraxisForCausalLM(PraxisConfig(**cfg)).eval()

    encoder = getattr(model, "encoder", None)
    aligned = getattr(encoder, "outputs_are_aligned", False) if encoder else False
    stub = types.SimpleNamespace(
        outputs_are_aligned=aligned,
        _compute_byte_nll_bits=BackpropagationTrainer._compute_byte_nll_bits,
    )

    ids = torch.randint(0, 256, (2, 128))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    labels = ids if aligned else ids[:, 1:].contiguous()

    bits = stub._compute_byte_nll_bits(stub, out, labels)
    assert bits is not None, f"{tag} emits no val_byte_nll_bits"

    # At random init the model is at chance, which for a V-way softmax is
    # log2(V). That is the whole reason this metric replaced the objective
    # rescaling: its LEVEL is interpretable against a known ceiling.
    chance = math.log2(out.logits.shape[-1])
    assert abs(float(bits) - chance) < 0.25, f"{tag}: {float(bits)} vs chance {chance}"

    # val_codec_bpb is the encoder-gated one, and none of these have a codec.
    assert not hasattr(encoder, "codec_recon_loss")
