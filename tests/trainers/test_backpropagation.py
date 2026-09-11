"""BackpropagationTrainer (praxis/trainers/backpropagation.py): the forward, the
Lightning train/validation step on a real packed byte batch, and the per-byte
NLL metric the validation step reports."""

import math
import time
import types

import pytest
import torch

from praxis import PraxisConfig, PraxisForCausalLM
from praxis.data.datasets.message_queue import MessageQueueManager
from praxis.optimization import get_optimizer, get_optimizer_profile
from praxis.schedulers import get_scheduler_func
from praxis.tokenizers import create_tokenizer
from praxis.trainers import BackpropagationTrainer


def test_trainer_forward_pass():
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
    optimizer_config, _ = get_optimizer_profile("AdamW")
    optimizer = get_optimizer(model, **optimizer_config)
    scheduler = get_scheduler_func(optimizer_config)(optimizer)
    trainer = BackpropagationTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        hparams={"batch_size": 4, "device": "cpu"},
        byte_level=False,
    )
    # A token-level model scores shifted labels.
    assert trainer.outputs_are_aligned is False

    outputs = trainer.forward(
        input_ids=torch.randint(0, 100, (2, 10)),
        labels=torch.randint(0, 100, (2, 9)),
    )
    assert torch.isfinite(outputs.loss)


class TestTrainingStepRunsEndToEnd:
    """The step itself on the dict WeightedIterableDataset yields, not just
    construction: a step that raises (a dropped local binding, a renamed batch
    key) fails every run while construction-only tests still pass."""

    @pytest.fixture
    def packed_batch(self):
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
        return batch, tokenizer

    @pytest.fixture
    def trainer(self, packed_batch):
        _, tokenizer = packed_batch
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
        return trainer

    def test_training_step_completes(self, trainer, packed_batch):
        batch, _ = packed_batch
        loss = trainer.training_step(batch, 0)
        assert torch.isfinite(loss)
        assert loss.requires_grad

    def test_validation_step_completes(self, trainer, packed_batch):
        batch, _ = packed_batch
        trainer.validation_step(batch, 0)

    def test_step_runs_without_the_optional_channels(self, trainer, packed_batch):
        """A bare tensor batch must still work: no block_ids, no masks."""
        batch, _ = packed_batch
        loss = trainer.training_step(batch["input_ids"], 0)
        assert torch.isfinite(loss)

    def test_packed_batch_is_pure_bytes_with_real_block_ids(self, packed_batch):
        """Guards the two halves of the pure-byte layout together."""
        batch, tokenizer = packed_batch
        assert tokenizer.byte_alphabet_size == 256
        assert int(batch["input_ids"].max()) < 256
        assert batch["block_ids"].shape == batch["input_ids"].shape
        # More than one document per row, or packing is not being exercised.
        assert len(set(batch["block_ids"][0].tolist())) > 1


# ── val_byte_nll_bits ────────────────────────────────────────────────────────
# `_compute_byte_nll_bits` only reads `self.outputs_are_aligned`, so a namespace
# stub stands in for the Lightning module.


def _byte_nll_bits(aligned, logits, labels):
    stub = types.SimpleNamespace(outputs_are_aligned=aligned)
    out = types.SimpleNamespace(logits=logits)
    return BackpropagationTrainer._compute_byte_nll_bits(stub, out, labels)


def test_byte_nll_bits_is_calibrated_against_chance():
    """Its LEVEL has to mean something: 8.0 for a uniform 256-way predictor,
    0.0 for a certain one, or it cannot be compared to a scaling law."""
    labels = torch.randint(0, 256, (2, 32))

    uniform = torch.zeros(2, 32, 256)
    assert abs(float(_byte_nll_bits(True, uniform, labels)) - 8.0) < 1e-4

    certain = torch.full((2, 32, 256), -1e4)
    certain.scatter_(-1, labels.unsqueeze(-1), 1e4)
    assert float(_byte_nll_bits(True, certain, labels)) < 1e-3

    # Padding must not be averaged in: masking half the targets to the ignore
    # index leaves a uniform predictor at 8.0, not pulled toward 0.
    masked = labels.clone()
    masked[:, ::2] = -100
    assert abs(float(_byte_nll_bits(True, uniform, masked)) - 8.0) < 1e-4

    # The codec-only helper is a plain nats-to-bits conversion.
    got = BackpropagationTrainer._compute_bits_per_byte(None, torch.tensor(3.0))
    assert abs(float(got) - 3.0 / math.log(2)) < 1e-6


def test_byte_nll_bits_shifts_for_unaligned_encoders():
    """An unaligned encoder's last position has no target, and the caller has
    already shifted the labels - so the logits must lose their last step. If the
    two conventions ever drift apart, the metric silently scores position t
    against byte t+1 and the number stops being comparable to anything."""
    labels = torch.randint(0, 256, (2, 31))
    logits = torch.zeros(2, 32, 256)

    # Unaligned: 32 logits, 31 labels - the trim makes them meet.
    assert abs(float(_byte_nll_bits(False, logits, labels)) - 8.0) < 1e-4

    # Aligned with a mismatched shape returns None rather than raising: a
    # missing series is readable, an exception inside validation is not.
    assert _byte_nll_bits(True, logits, labels) is None


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
    """Any byte-level tokenizer gets this metric, not just codec encoders.

    The gate is `self.byte_level`, which comes from the TOKENIZER
    (`cli/config.py`: `tokenizer_type == "byte_level"`), and the emission sits
    outside the codec branch - so a vanilla byte-latent run with no codec, and a
    plain model with no encoder at all, both get it. Only `val_codec_bpb` is
    encoder-gated. It leads METRIC_PRIORITY, so a family that stops emitting it
    drops out of every run comparison.
    """
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

    ids = torch.randint(0, 256, (2, 128))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    labels = ids if aligned else ids[:, 1:].contiguous()

    bits = _byte_nll_bits(aligned, out.logits, labels)
    assert bits is not None, f"{tag} emits no val_byte_nll_bits"

    # At random init the model is at chance, which for a V-way softmax is log2(V).
    chance = math.log2(out.logits.shape[-1])
    assert abs(float(bits) - chance) < 0.25, f"{tag}: {float(bits)} vs chance {chance}"

    # val_codec_bpb is the encoder-gated one, and none of these have a codec.
    assert not hasattr(encoder, "codec_recon_loss")
