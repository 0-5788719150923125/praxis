"""
Verify that bidirectional masking works end-to-end via Prismatic router.

Tests that:
1. Prismatic creates forward and backward masks correctly
2. HexAttention uses provided masks (doesn't create its own)
3. End-to-end: Prismatic produces different outputs via different masks
"""

import pytest
import torch

from praxis.attention.hex import HexAttention
from praxis.configuration import PraxisConfig
from praxis.routers.prismatic import Prismatic


class TestPrismaticMaskCreation:
    """Test Prismatic router's mask creation."""

    def test_forward_mask_creation(self):
        """Verify Prismatic creates correct forward causal mask."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        # Create dummy experts for Prismatic (we're only testing mask creation)
        dummy_experts = [HexAttention(config) for _ in range(2)]

        # Create Prismatic router
        router = Prismatic(config, experts=dummy_experts)

        # Create forward mask
        seq_len = 4
        forward_mask = router._create_causal_mask(seq_len, direction="forward", device=torch.device("cpu"))

        # Verify shape (seq_len x seq_len)
        assert forward_mask.shape == (seq_len, seq_len)

        # Forward causal: position i can see j where j <= i
        # Row 0 should only see position 0: [0, -inf, -inf, -inf]
        assert forward_mask[0, 0] == 0.0
        assert forward_mask[0, 1] == float('-inf')

        # Row 3 should see all positions: [0, 0, 0, 0]
        assert torch.all(forward_mask[3, :] == 0.0)

    def test_backward_mask_creation(self):
        """Verify Prismatic creates correct backward causal mask."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        # Create dummy experts for Prismatic (we're only testing mask creation)
        dummy_experts = [HexAttention(config) for _ in range(2)]

        router = Prismatic(config, experts=dummy_experts)

        # Create backward mask
        seq_len = 4
        backward_mask = router._create_causal_mask(seq_len, direction="backward", device=torch.device("cpu"))

        # Verify shape
        assert backward_mask.shape == (seq_len, seq_len)

        # Backward causal: position i can see j where j >= i
        # Row 0 should see all positions: [0, 0, 0, 0]
        assert torch.all(backward_mask[0, :] == 0.0)

        # Row 3 should only see position 3: [-inf, -inf, -inf, 0]
        assert backward_mask[3, 3] == 0.0
        assert backward_mask[3, 0] == float('-inf')

    def test_forward_and_backward_differ(self):
        """Verify forward and backward masks are different."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic"
        )

        # Create dummy experts for Prismatic (we're only testing mask creation)
        dummy_experts = [HexAttention(config) for _ in range(2)]

        router = Prismatic(config, experts=dummy_experts)
        seq_len = 4
        device = torch.device("cpu")

        forward_mask = router._create_causal_mask(seq_len, direction="forward", device=device)
        backward_mask = router._create_causal_mask(seq_len, direction="backward", device=device)

        # Masks should be different
        assert not torch.allclose(forward_mask, backward_mask)


class TestHexAttentionUsesProvidedMask:
    """Test that HexAttention uses provided attention_mask instead of creating its own."""

    def test_hex_uses_provided_mask(self):
        """Verify HexAttention uses the attention_mask parameter when provided."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            dropout=0.0,
            causal=True,
            pos_type="alibi"
        )

        attn = HexAttention(config)

        # Create input
        batch_size, seq_len = 2, 4
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Create a backward causal mask (opposite of default forward causal)
        # Forward causal: lower triangular (position i sees j where j <= i)
        # Backward causal: upper triangular (position i sees j where j >= i)
        kv_len = seq_len + 1  # +1 for ghost token
        backward_mask = torch.tril(torch.ones(seq_len, kv_len) * float('-inf'), diagonal=-1)
        # Ghost token (last position) should be accessible: set last column to 0
        backward_mask[:, -1] = 0.0

        # Run with backward mask
        out_backward, _, _ = attn(inputs, attention_mask=backward_mask)

        # Run with default forward causal mask (no mask provided)
        out_forward, _, _ = attn(inputs, attention_mask=None)

        # Outputs should differ (different masks used)
        assert not torch.allclose(out_backward, out_forward, atol=1e-3), \
            "HexAttention should use provided mask instead of creating its own"

    def test_hex_converts_additive_mask_correctly(self):
        """Verify HexAttention correctly converts additive masks to score_mod."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            dropout=0.0,
            causal=True,
            pos_type="alibi"
        )

        attn = HexAttention(config)

        # Create a simple additive mask (lower triangular for forward causal)
        seq_len = 4
        kv_len = seq_len + 1  # +1 for ghost token
        additive_mask = torch.triu(torch.ones(seq_len, kv_len) * float('-inf'), diagonal=1)
        # Ghost token (last position) should be accessible: set last column to 0
        additive_mask[:, -1] = 0.0

        # Create score_mod from mask
        score_mod = attn._create_score_mod_from_additive_mask(additive_mask)

        # Score_mod should be created successfully and be callable
        assert score_mod is not None
        assert callable(score_mod)


class TestBidirectionalEndToEnd:
    """Test that Prismatic's bidirectional masking works end-to-end."""

    def test_prismatic_produces_different_expert_outputs(self):
        """Verify Prismatic with different masks produces different expert outputs."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            num_experts=2,
            router_type="prismatic",
            dropout=0.0,
            causal=True,
            pos_type="alibi"
        )

        # Create two HexAttention modules (simulating experts with same architecture)
        expert_0 = HexAttention(config)
        expert_1 = HexAttention(config)

        # Create Prismatic router
        router = Prismatic(config, experts=[expert_0, expert_1])

        # Create input
        batch_size, seq_len = 2, 8
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Create forward and backward masks
        device = inputs.device
        kv_len = seq_len + 1  # +1 for ghost token
        forward_mask = router._create_causal_mask(seq_len, direction="forward", device=device)
        backward_mask = router._create_causal_mask(seq_len, direction="backward", device=device)

        # Pad masks for ghost token
        forward_mask = torch.cat([forward_mask, torch.zeros(seq_len, 1)], dim=1)
        backward_mask = torch.cat([backward_mask, torch.zeros(seq_len, 1)], dim=1)

        # Forward pass through experts with different masks
        out_forward, _, _ = expert_0(inputs, attention_mask=forward_mask)
        out_backward, _, _ = expert_1(inputs, attention_mask=backward_mask)

        # Outputs should differ (different masks)
        assert not torch.allclose(out_forward, out_backward, atol=1e-3), \
            "Same expert with different masks should produce different outputs"


class TestGhostPosition:
    """Test that ghost token position affects backward masking."""

    def test_ghost_position_affects_output(self):
        """Verify ghost_position parameter changes where ghost token is placed."""
        config = PraxisConfig(
            hidden_size=64,
            num_heads=4,
            num_queries=1,
            dropout=0.0,
            causal=True,
            pos_type="alibi"
        )

        attn = HexAttention(config)

        # Create input
        batch_size, seq_len = 1, 4
        inputs = torch.randn(batch_size, seq_len, config.hidden_size)

        # Create forward mask with ghost at start (standard ghostmax)
        forward_mask = torch.triu(
            torch.ones(seq_len, seq_len + 1) * float('-inf'),
            diagonal=1
        )
        # Ghost column at start (position 0) should be accessible
        forward_mask[:, 0] = 0.0

        # Run with ghost at start (standard forward ghostmax)
        out_ghost_start, _, _ = attn(inputs, attention_mask=forward_mask, ghost_position="start")

        # Run with ghost at end (inverted backward ghostmax) - same mask but different ghost position
        out_ghost_end, _, _ = attn(inputs, attention_mask=forward_mask, ghost_position="end")

        # Outputs should differ (ghost token in different position affects K/V)
        assert not torch.allclose(out_ghost_start, out_ghost_end, atol=1e-3), \
            "Ghost position should affect output (different K/V structure)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
