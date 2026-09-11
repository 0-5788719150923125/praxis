import pytest
import torch
import torch.nn as nn
from torch import nn

from praxis import registry
from praxis.encoders.byte_latent.encoder import (
    create_patch_block_ids,
    mask_entropy_preds_at_special_tokens,
    packed_rnn_block,
    pooling_downsample,
)
from praxis.encoders.byte_latent.merge import MERGES, PatchMerge

# ------------------------------------------------------------------------------
# byte_merge
# ------------------------------------------------------------------------------
# The local decoder's byte/trunk merge, and the in-context copy probe that says whether
# the trunk's long-range signal reaches the output at all.


DIM = 16


def test_add_is_the_blt_sum():
    merge = PatchMerge(DIM, "add")
    byte, trunk = torch.randn(2, 5, DIM), torch.randn(2, 5, DIM)
    torch.testing.assert_close(merge(byte, trunk), byte + trunk)


def test_gated_starts_at_an_even_split_of_normalized_streams():
    merge = PatchMerge(DIM, "gated").train()
    byte, trunk = torch.randn(2, 5, DIM) * 300, torch.randn(2, 5, DIM)
    out = merge(byte, trunk)
    norm = lambda x: torch.nn.functional.rms_norm(x, (DIM,), eps=1e-6)
    torch.testing.assert_close(out, 0.5 * norm(byte) + 0.5 * norm(trunk))
    metrics = merge.training_metrics()
    assert metrics["merge_gate_trunk"] == pytest.approx(0.5)
    assert metrics["merge_trunk_ratio"] == pytest.approx(1.0, rel=1e-4)


def test_gated_share_cannot_be_won_by_scale():
    """Gated, scaling either stream changes nothing."""
    merge = PatchMerge(DIM, "gated")
    with torch.no_grad():
        nn.init.normal_(merge.gate.weight)
    byte, trunk = torch.randn(2, 5, DIM), torch.randn(2, 5, DIM)
    torch.testing.assert_close(merge(byte * 100, trunk), merge(byte, trunk))
    torch.testing.assert_close(merge(byte, trunk * 100), merge(byte, trunk))


def test_gated_merge_reads_only_its_own_position():
    merge = PatchMerge(DIM, "gated")
    with torch.no_grad():
        nn.init.normal_(merge.gate.weight)
    byte, trunk = torch.randn(1, 6, DIM), torch.randn(1, 6, DIM)
    edited = trunk.clone()
    edited[0, 4] += torch.randn(DIM)
    a, b = merge(byte, trunk), merge(byte, edited)
    torch.testing.assert_close(a[0, :4], b[0, :4], rtol=0.0, atol=0.0)
    torch.testing.assert_close(a[0, 5:], b[0, 5:], rtol=0.0, atol=0.0)


def test_both_modes_report_the_magnitudes_and_only_gated_reports_a_gate():
    for mode in MERGES:
        merge = PatchMerge(DIM, mode).train()
        merge(torch.randn(2, 5, DIM), torch.randn(2, 5, DIM))
        keys = set(merge.training_metrics())
        assert {"merge_trunk_ratio", "merge_trunk_content_ratio"} <= keys
        assert ("merge_gate_trunk" in keys) == (mode == "gated")
        assert keys <= set(PatchMerge.metric_descriptions)


def test_an_unknown_merge_is_refused():
    with pytest.raises(ValueError, match="Unknown merge"):
        PatchMerge(DIM, "concat")


def test_the_unimplemented_cross_attention_decoder_is_refused():
    from praxis import PraxisConfig
    from praxis.encoders.byte_latent.encoder import ByteLatentEncoder

    config = PraxisConfig(hidden_size=32, embed_size=32, vocab_size=1024)
    with pytest.raises(NotImplementedError, match="cross_attn_decoder"):
        ByteLatentEncoder(config, cross_attn_decoder=True)


# ------------------------------------------------------------------------------
# encoders
# ------------------------------------------------------------------------------


def test_create_patch_block_ids():
    """Test patch block ID creation with special tokens."""
    device = "cpu"
    batch_size = 2
    seq_len = 8
    num_patches = 4  # each patch is size 2

    # Create sample input with special tokens (0) at specific positions
    input_ids = torch.tensor(
        [
            [1, 2, 0, 4, 5, 0, 7, 8],  # Two special tokens
            [1, 0, 3, 4, 0, 6, 0, 8],  # Three special tokens
        ],
        device=device,
    )

    # Create patch lengths - each patch is size 2
    patch_lengths = torch.full(
        (batch_size, num_patches), 2, device=device, dtype=torch.long
    )

    # Create patch IDs - each position maps to its patch (0-3)
    patch_ids = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 0, 1, 1, 2, 2, 3, 3],
        ],
        device=device,
        dtype=torch.long,
    )

    # Get block IDs
    block_ids = create_patch_block_ids(
        input_ids=input_ids,
        patch_lengths=patch_lengths,
        patch_ids=patch_ids,
        special_tokens=[0],
    )

    # Updated expected output to be patch-level
    expected = torch.tensor(
        [
            [1, 1, 2, 3],  # 4 patches for first sequence
            [1, 2, 2, 3],  # 4 patches for second sequence
        ],
        device=device,
        dtype=torch.long,
    )

    assert block_ids.shape == expected.shape
    assert torch.all(
        block_ids == expected
    ), f"Block IDs mismatch.\nGot:      {block_ids}\nExpected: {expected}"

    # Test edge case: all special tokens
    input_ids_all_special = torch.zeros((1, seq_len), device=device, dtype=torch.long)
    block_ids_all_special = create_patch_block_ids(
        input_ids=input_ids_all_special,
        patch_lengths=patch_lengths[0:1],
        patch_ids=patch_ids[0:1],
        special_tokens=[0],
    )

    expected_all_special = torch.tensor(
        [[1, 2, 3, 4]],  # 4 patches, each containing special tokens
        device=device,
        dtype=torch.long,
    )
    assert block_ids_all_special.shape == expected_all_special.shape
    assert torch.all(
        block_ids_all_special == expected_all_special
    ), f"All special tokens case mismatch.\nGot:      {block_ids_all_special}\nExpected: {expected_all_special}"


def test_mask_entropy_preds_at_special_tokens():

    # Test 1: Basic test with small tensors
    print("=== Test 1: Basic Test ===")
    # Create a small batch of input_ids with some special tokens (0)
    input_ids = torch.tensor(
        [
            [1, 2, 0, 4],  # First sequence has a special token at position 2
            [5, 0, 7, 8],  # Second sequence has a special token at position 1
        ]
    )

    # Create entropy_preds with a small vocab size (3)
    # For simplicity, fill with increasing values starting from 1 to avoid zeros
    vocab_size = 3
    seq_len = input_ids.shape[1]
    batch_size = input_ids.shape[0]

    # Create entropy_preds as a flattened tensor [batch_size, seq_len * vocab_size]
    entropy_preds = torch.arange(
        1, batch_size * seq_len * vocab_size + 1, dtype=torch.float32
    )
    entropy_preds = entropy_preds.reshape(batch_size, seq_len * vocab_size)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs:\n{input_ids}")

    print(f"Original entropy_preds shape: {entropy_preds.shape}")
    print(f"Original entropy_preds (flattened):\n{entropy_preds}")

    # Reshape to show the logical 3D structure
    print("Original entropy_preds (reshaped to 3D for visualization):")
    print(entropy_preds.reshape(batch_size, seq_len, vocab_size))

    # Make a copy of the original for comparison
    original_3d = entropy_preds.clone().reshape(batch_size, seq_len, vocab_size)

    # Apply masking
    masked_preds = mask_entropy_preds_at_special_tokens(
        input_ids, entropy_preds, special_tokens=[0]
    )

    print(f"Masked entropy_preds shape: {masked_preds.shape}")
    print("Masked entropy_preds (reshaped to 3D for visualization):")
    print(masked_preds.reshape(batch_size, seq_len, vocab_size))

    # Verify that predictions at special token positions are zeroed out
    # Reshape for easier verification
    masked_3d = masked_preds.reshape(batch_size, seq_len, vocab_size)

    # Check first batch, position 2 (should be zeros)
    assert torch.all(
        masked_3d[0, 2, :] == 0
    ), "Special token position not correctly masked in batch 0"

    # Check second batch, position 1 (should be zeros)
    assert torch.all(
        masked_3d[1, 1, :] == 0
    ), "Special token position not correctly masked in batch 1"

    # Check that non-special token positions are unchanged from original
    assert torch.all(
        masked_3d[0, 0, :] == original_3d[0, 0, :]
    ), "Non-special token position incorrectly modified in batch 0"
    assert torch.all(
        masked_3d[0, 1, :] == original_3d[0, 1, :]
    ), "Non-special token position incorrectly modified in batch 0"
    assert torch.all(
        masked_3d[0, 3, :] == original_3d[0, 3, :]
    ), "Non-special token position incorrectly modified in batch 0"

    print("All assertions passed for Test 1!")

    # Test 2: More realistic dimensions
    print("\n=== Test 2: Realistic Dimensions ===")
    # Create input_ids with more realistic dimensions
    batch_size = 2
    seq_len = 64
    vocab_size = 260  # Similar to your actual case

    # Create random input_ids with some special tokens (0)
    input_ids = torch.randint(1, 10, (batch_size, seq_len))
    # Randomly place special tokens
    special_positions = torch.randint(
        0, seq_len, (batch_size, 5)
    )  # 5 special tokens per batch

    for batch_idx in range(batch_size):
        for pos in special_positions[batch_idx]:
            input_ids[batch_idx, pos] = 0  # Set special token

    # Create random entropy_preds
    entropy_preds = torch.rand(batch_size, seq_len * vocab_size)

    # Make a copy of the original for comparison
    original_3d = entropy_preds.clone().reshape(batch_size, seq_len, vocab_size)

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Special token positions in batch 0: {torch.where(input_ids[0] == 0)[0]}")
    print(f"Special token positions in batch 1: {torch.where(input_ids[1] == 0)[0]}")

    print(f"Original entropy_preds shape: {entropy_preds.shape}")

    # Apply masking
    masked_preds = mask_entropy_preds_at_special_tokens(
        input_ids, entropy_preds, special_tokens=[0]
    )

    print(f"Masked entropy_preds shape: {masked_preds.shape}")

    # Reshape for verification
    masked_3d = masked_preds.reshape(batch_size, seq_len, vocab_size)

    # Verify masking for special tokens in batch 0
    for pos in torch.where(input_ids[0] == 0)[0]:
        assert torch.all(
            masked_3d[0, pos, :] == 0
        ), f"Special token at position {pos} not masked in batch 0"

    # Verify masking for special tokens in batch 1
    for pos in torch.where(input_ids[1] == 0)[0]:
        assert torch.all(
            masked_3d[1, pos, :] == 0
        ), f"Special token at position {pos} not masked in batch 1"

    # Verify that non-special token positions are unchanged
    for batch_idx in range(batch_size):
        for pos in range(seq_len):
            if input_ids[batch_idx, pos] != 0:  # If not a special token
                assert torch.all(
                    masked_3d[batch_idx, pos, :] == original_3d[batch_idx, pos, :]
                ), f"Non-special token at position {pos} incorrectly modified in batch {batch_idx}"

    # Test that the masked entropy_preds has the same shape as the original
    assert (
        masked_preds.shape == entropy_preds.shape
    ), "Shape mismatch between original and masked predictions"

    print("All assertions passed for Test 2!")


def test_packed_rnn():
    """Every packed document is processed, each with a fresh hidden state -
    positions after an EOS are never dropped or zeroed."""
    batch_size, seq_len, feature_dim = 2, 5, 3
    hidden_dim = 4

    x = torch.randn(batch_size, seq_len, feature_dim)
    input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

    input_ids[0, 2] = 0  # First row: document boundary at position 2
    # Second row: no EOS, so one document spanning the whole row.

    rnn = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
    output = packed_rnn_block(rnn, x, input_ids, eos_token_id=0)

    assert output.shape == (batch_size, seq_len, hidden_dim)

    # The document AFTER the boundary is real work, not padding.
    assert output[0, 3:].abs().mean().item() > 1e-5
    assert output[1].abs().mean().item() > 1e-5

    # Each document is an independent sequence: block 1 is positions 0..2
    # (create_block_ids increments AFTER the EOS, so the EOS closes its own
    # block), block 2 is positions 3..4, and each starts from a zero state.
    first, _ = rnn(x[0:1, :3])
    second, _ = rnn(x[0:1, 3:])
    assert torch.allclose(output[0:1, :3], first, atol=1e-5)
    assert torch.allclose(output[0:1, 3:], second, atol=1e-5)

    # A row with no boundary must be bit-identical to a plain full-length run,
    # which is what keeps `chat_format: default` runs unchanged.
    whole, _ = rnn(x[1:2])
    assert torch.allclose(output[1:2], whole, atol=1e-5)


def test_packed_rnn_matches_explicit_block_ids():
    """The derived-from-EOS path and the passed-in block_ids path agree."""
    from praxis.utils import create_block_ids

    x = torch.randn(2, 6, 3)
    input_ids = torch.ones(2, 6, dtype=torch.long)
    input_ids[0, 1] = 0
    input_ids[0, 4] = 0

    rnn = nn.LSTM(3, 4, batch_first=True)
    derived = packed_rnn_block(rnn, x, input_ids, eos_token_id=0)
    explicit = packed_rnn_block(
        rnn, x, input_ids, eos_token_id=0, block_ids=create_block_ids(input_ids, 0)
    )
    assert torch.allclose(derived, explicit, atol=1e-6)


def test_topk_mean_pooling():
    """Test topk_mean_pooling with more realistic data."""
    # More realistic dimensions
    batch_size = 2
    seq_len = 12
    emb_dim = 8
    max_num_patches = 4
    k = 3

    # Create input with varied embeddings
    h = torch.randn(batch_size, seq_len, emb_dim) * 5  # Random values, scaled up

    # Create patches of varying sizes
    patch_ids = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3],  # Sizes: 3,4,2,3
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3],  # Sizes: 2,4,3,3
        ],
        dtype=torch.long,
    )

    # Call function
    result = pooling_downsample(h, max_num_patches, f"topk:{k}", patch_ids)

    # Manually calculate expected results
    expected = torch.zeros(batch_size, max_num_patches, emb_dim)

    # Verify for each batch and patch
    for b in range(batch_size):
        for p in range(max_num_patches):
            # Get values for this patch
            patch_mask = patch_ids[b] == p
            patch_vals = h[b][patch_mask]

            # Calculate expected top-k mean
            if len(patch_vals) > 0:
                k_actual = min(k, len(patch_vals))
                topk_vals, _ = torch.topk(patch_vals, k_actual, dim=0)
                expected[b, p] = topk_vals.mean(dim=0)

    # Verify results
    assert torch.allclose(result, expected, rtol=1e-5), (
        f"Mismatch in topk_mean_pooling results.\n"
        f"Got:\n{result}\n"
        f"Expected:\n{expected}"
    )

    # Add edge case tests
    # Test with k=1 (max pooling equivalent)
    result_k1 = pooling_downsample(h, max_num_patches, "max", patch_ids=patch_ids)
    # Test with k=seq_len (mean pooling equivalent)
    result_kmax = pooling_downsample(
        h, max_num_patches, f"topk:{k}", patch_ids=patch_ids
    )

    # Verify shapes
    assert result.shape == (batch_size, max_num_patches, emb_dim)
    assert result_k1.shape == (batch_size, max_num_patches, emb_dim)
    assert result_kmax.shape == (batch_size, max_num_patches, emb_dim)

    # Verify difference
    # Get results for all pooling modes
    result_max = pooling_downsample(h, max_num_patches, "max", patch_ids=patch_ids)
    result_min = pooling_downsample(h, max_num_patches, "min", patch_ids=patch_ids)
    result_mean = pooling_downsample(h, max_num_patches, "avg", patch_ids=patch_ids)
    result_topk = pooling_downsample(
        h, max_num_patches, f"topk:{k}", patch_ids=patch_ids
    )

    # Verify all results are different
    assert not torch.allclose(
        result_max, result_topk
    ), "topk_mean should differ from max pooling"
    assert not torch.allclose(
        result_min, result_topk
    ), "topk_mean should differ from min pooling"
    assert not torch.allclose(
        result_mean, result_topk
    ), "topk_mean should differ from mean pooling"

    # Additional verification that results make sense
    # topk_mean should be between max and min
    assert torch.all(
        result_topk <= result_max
    ), "topk_mean should not exceed maximum values"
    assert torch.all(
        result_topk >= result_min
    ), "topk_mean should not be less than minimum values"


# ------------------------------------------------------- static patching


class TestStaticPatchingNeedsNoBOE:
    """Fixed-size patches on a uniform lattice, with no injected tokens.

    The BLT reference keeps static patches uniform and prepends `patch_size-1`
    BOE tokens so the first patch holds exactly one real byte (the decoder lag,
    asserted by `decoder_patch_ids_from_lengths`). Under a 256-byte alphabet
    there is no spare id, so those would be literal 0x00 bytes at the head of
    every sequence. Emitting `[1, P, P, ...]` satisfies the same lag with
    nothing added to the input.
    """

    @staticmethod
    def _encoder(mode, patch_size=8):
        from praxis import PraxisConfig

        name = (
            "abstractinator_v1"
            if mode == "static"
            else "abstractinator_harmonic_gdn_vocab_bank"
        )
        config = PraxisConfig(
            vocab_size=1024,
            byte_vocab_size=256,
            byte_offset=0,
            hidden_size=111,
            embed_size=110,
            num_heads=1,
            head_size=30,
            depth=2,
            num_layers=1,
            encoder_type=name,
            decoder_type="sequential",
            block_size=64,
            max_position_embeddings=1024,
            device_map="cpu",
        )
        return registry.lookup("encoders", name)(config)

    def test_static_prepends_no_boe(self):
        encoder = self._encoder("static")
        assert encoder.byte_config.patching_mode == "static"
        assert encoder.byte_config.patch_size == 8
        assert encoder.nb_boe == 0, "static patching must not inject BOE bytes"

    def test_space_also_needs_no_boe(self):
        assert self._encoder("space").nb_boe == 0

    def test_entropy_still_gets_its_prefix(self):
        """Entropy boundaries follow the model, so a length-1 first patch is
        not guaranteed and the prefix is still doing real work."""
        encoder = self._encoder("static")
        encoder.byte_config.patching_mode = "entropy"
        assert encoder.nb_boe == encoder.byte_config.patch_size - 1

    @pytest.mark.parametrize("patch_size", [4, 6, 8])
    @pytest.mark.parametrize("seq_len", [31, 64, 65])
    def test_lengths_are_one_then_uniform(self, patch_size, seq_len):
        from praxis.encoders.byte_latent.patcher import (
            Patcher,
            PatcherConfig,
            PatchingMode,
        )

        patcher = Patcher(
            PatcherConfig(
                patching_mode=PatchingMode.static,
                patch_size=patch_size,
                device="cpu",
                byte_offset=0,
            )
        )
        tokens = torch.randint(0, 256, (3, seq_len))
        lengths, _ = patcher.patch(tokens, include_next_token=True)

        # The lag contract decoder_patch_ids_from_lengths asserts.
        assert torch.all(lengths[:, 0] == 1)
        # Every patch but the first and last is exactly patch_size.
        if lengths.shape[1] > 2:
            assert torch.all(lengths[:, 1:-1] == patch_size)
        # Total must cover the sequence plus the next-token slot.
        assert torch.all(lengths.sum(dim=1) == seq_len + 1)
        assert torch.all(lengths > 0)


class TestEntropyBoundaryRulesAreAlternatives:
    """BLT's two entropy rules are alternatives, not a union.

    global    H(x_t) > theta_g
    monotonic H(x_t) - H(x_t-1) > theta_r

    OR-ing them is neither rule and cuts strictly more often than either.
    Inert for space/static runs, but the deviation is from a paper we cite.
    """

    @staticmethod
    def _patch(monotonicity, entropies, threshold=1.0):
        from praxis.encoders.byte_latent.patcher import (
            Patcher,
            PatcherConfig,
            PatchingMode,
        )

        patcher = Patcher(
            PatcherConfig(
                patching_mode=PatchingMode.entropy,
                device="cpu",
                monotonicity=monotonicity,
                byte_offset=0,
            )
        )
        tokens = torch.zeros(entropies.shape, dtype=torch.long)
        lengths, _ = patcher.patch(
            tokens, include_next_token=False, threshold=threshold, entropies=entropies
        )
        return lengths

    def test_monotonic_ignores_absolute_level(self):
        """A flat, uniformly HIGH entropy run has no jumps, so the monotonic
        rule must not cut it - while the global rule cuts everywhere. Under the
        old union the monotonic setting inherited the global cuts."""
        entropies = torch.full((1, 24), 10.0)
        mono = self._patch(True, entropies)
        glob = self._patch(False, entropies)
        assert (mono > 0).sum() < (
            glob > 0
        ).sum(), "monotonic rule is still inheriting the absolute rule's boundaries"

    def test_monotonic_cuts_on_a_jump(self):
        """A single upward step is exactly what the monotonic rule is for."""
        entropies = torch.zeros(1, 16)
        entropies[0, 8:] = 5.0
        lengths = self._patch(True, entropies)
        assert int((lengths > 0).sum()) >= 2, "no boundary at the entropy jump"

    def test_both_rules_conserve_the_sequence(self):
        entropies = torch.rand(2, 32) * 4
        for monotonicity in (True, False):
            lengths = self._patch(monotonicity, entropies)
            assert torch.all(lengths.sum(dim=1) == 32)


# ------------------------------------------------------------------------------
# chat_formats
# ------------------------------------------------------------------------------
# Tests for the ``chat_formats`` registry and the text-boundary (prose) format.
#
# The invariants worth pinning are the ones that silently produce a broken run rather
# than an exception:
#
# - the `default` profile must stay byte-identical, since every existing checkpoint's
# data pipeline depends on it, - the boundary that ENDS a generated turn must be a
# trained target (the defect `prose` exists to remove), - a stop-string halt must not
# re-fire on the boundary it resumed from, or the tool loop returns zero new tokens
# forever, - the tool flow's three boundaries must classify unambiguously.


# ---------------------------------------------------------- patch budget


def test_prose_boundary_costs_fewer_patches(prose_tokenizer):
    """A control token cuts a patch unconditionally (the `|= tokens < OFFSET`
    in find_space_patch_start_ids runs after the run-collapse), so each one
    buys its own patch. Text boundaries fold into the newline run."""
    from praxis.encoders.byte_latent.patcher import (
        find_space_patch_start_ids,
        patch_lengths_from_start_ids,
    )

    def patch_count(text):
        ids = prose_tokenizer.encode(text)
        t = torch.tensor([ids])
        lengths = patch_lengths_from_start_ids(
            find_space_patch_start_ids(t), t.shape[1]
        )
        return int((lengths[0] > 0).sum())

    assert patch_count("France.\n\nuser\n\nAnd of Japan?") < patch_count(
        "France.\n[SEP]\n[BOS]user\nAnd of Japan?"
    )
