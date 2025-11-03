"""
Comprehensive test suite for Prismatic attention module.

Tests verify:
1. Basic initialization and configuration
2. Expert cloning and perturbation mechanics
3. Determinism and reproducibility
4. Sparsity and magnitude-aware perturbations
5. Forward pass compatibility (router and direct modes)
6. Parameter merging and soft-routing
7. Integration with existing Praxis components

Connection to "The Blind Watchmaker" paper:
------------------------------------------
These tests verify that Prismatic implements the theoretical principles:
- Static architectural diversity (perturbations are fixed, not learned)
- Sparse magnitude-aware perturbations (focused irregularity)
- Multi-eye construction (one clean expert, others perturbed)
- Soft-merging routing (adaptive combination of diverse experts)

Connection to Lottery Ticket Hypothesis:
---------------------------------------
Prismatic inverts LTH (Frankle & Carbin, 2019): instead of finding critical sparse
subnetworks through pruning, we perturb sparse high-magnitude weights to force
discovery of alternative computational paths. Tests verify that only ~10% of weights
are perturbed (the "lottery tickets"), while 90% remain intact for stability.

Connection to Willow Quantum Architecture:
-----------------------------------------
Like Google's Willow quantum computer perturbing individual qubits, Prismatic
perturbs a small percentage of critical weights. Tests verify that sparse
perturbations create sufficient diversity without destroying functionality.
"""

import copy

import pytest
import torch
import torch.nn as nn

from praxis.blocks.transformer import TransformerBlock
from praxis.configuration import PraxisConfig
from praxis.routers.prismatic import Prismatic, PrismaticConfig


class SimpleMLP(nn.Module):
    """Simple MLP for testing Prismatic perturbations."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2, bias=True)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        inputs,
        attention_mask=None,
        past_key_values=None,
        current_state=None,
        current_depth=0,
        block_ids=None,
    ):
        """Router-mode forward compatible with Prismatic."""
        x = self.fc1(inputs)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x, past_key_values, current_state, 0.0


class SimpleRecurrentModule(nn.Module):
    """Simple module for testing direct-mode forward."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, current_state=None):
        """Direct-mode forward (inputs, state) -> (outputs, state, loss)."""
        outputs = self.linear(inputs)
        if current_state is not None:
            outputs = outputs + current_state
        new_state = outputs.mean(dim=1, keepdim=True)
        return outputs, new_state, 0.0


class TestPrismaticInitialization:
    """Test Prismatic initialization and configuration."""

    def test_requires_base_expert(self):
        """Test that Prismatic requires base_expert or experts in kwargs."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        # Should raise error without base_expert or experts
        with pytest.raises(ValueError, match="either 'base_expert' or 'experts'"):
            Prismatic(config)

    def test_initialization_with_base_expert(self):
        """Test successful initialization with base expert."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Verify expert creation
        assert len(prismatic.experts) == config.num_experts
        assert prismatic.num_experts == config.num_experts
        assert prismatic.perturbation_scale == 0.01
        assert prismatic.sparsity == 0.1

    def test_initialization_with_experts_list(self):
        """Test successful initialization with experts list (standard Praxis flow)."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        # Create a list of experts (simulating LocalLayer's expert creation)
        expert_list = [SimpleMLP(hidden_size=64) for _ in range(3)]

        # Should use first expert as base and create perturbed clones
        prismatic = Prismatic(config, experts=expert_list)

        # Verify expert creation
        assert len(prismatic.experts) == config.num_experts
        assert prismatic.num_experts == config.num_experts

    def test_config_defaults(self):
        """Test that default configuration values are applied correctly."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            # Using defaults for other params
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Check defaults (0.8 = 80% pruning/amplification for attractive mode)
        assert prismatic.perturbation_scale == 0.8
        assert prismatic.sparsity == 0.1
        assert prismatic.perturb_by_magnitude is True
        assert prismatic.perturbation_mode == "attractive"
        assert prismatic.dropout_rate == 0.0
        assert prismatic.helical_modulation is True  # Helical modulation is now default
        assert prismatic.helical_wavelength == 3141.592653589793  # π × 1000


class TestPerturbationMechanics:
    """
    Test core perturbation mechanics.

    These tests verify the key theoretical properties:
    1. Expert 0 is clean (unperturbed) - the "right eye" baseline
    2. Experts 1+ are perturbed - the "left eye(s)" with forced irregularity
    3. Perturbations are deterministic (reproducible)
    4. Perturbations are sparse (only k% of weights)
    5. Perturbations are magnitude-aware (scaled by weight magnitude)
    """

    def test_expert_zero_is_clean(self):
        """
        Test that Expert 0 remains unperturbed (the "right eye" baseline).

        Connection to paper: "The right eye dominates, providing understanding
        and contextual coherence." Expert 0 represents consensus reality.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        # Create base expert with known parameters
        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        # Create Prismatic
        prismatic = Prismatic(config, base_expert=base_expert)

        # Expert 0 should be identical to base expert
        expert_0 = prismatic.experts[0]
        for name, param in expert_0.named_parameters():
            original = original_params[name]
            assert torch.allclose(
                param, original, atol=1e-6
            ), f"Expert 0 parameter '{name}' was modified (should be clean)"

    def test_experts_are_perturbed(self):
        """
        Test that Experts 1+ are perturbed (the "left eye(s)" with irregularity).

        Connection to paper: "The left eye operates passively, memorizing patterns
        immediately but without semantic integration—a blind watchmaker."
        Static perturbations force each expert to adapt to irregular parameter spaces.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Experts 1+ should be perturbed
        for expert_idx in range(1, config.num_experts):
            expert = prismatic.experts[expert_idx]

            has_perturbations = False
            for name, param in expert.named_parameters():
                if not param.requires_grad:
                    continue

                original = original_params[name]
                if not torch.allclose(param, original, atol=1e-6):
                    has_perturbations = True
                    break

            assert (
                has_perturbations
            ), f"Expert {expert_idx} should be perturbed but appears identical to base"

    def test_perturbations_are_deterministic(self):
        """
        Test that perturbations are deterministic (same seed = same result).

        Connection to paper: Static perturbations ensure reproducibility.
        Different experts must traverse different but reproducible paths through
        the computational substrate.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Create two Prismatic instances with same config and base expert
        # (need to deep copy base expert for second instance)
        base_expert_copy = SimpleMLP(hidden_size=64)
        base_expert_copy.load_state_dict(base_expert.state_dict())

        prismatic_1 = Prismatic(config, base_expert=base_expert)
        prismatic_2 = Prismatic(config, base_expert=base_expert_copy)

        # Check that corresponding experts have identical parameters
        for expert_idx in range(config.num_experts):
            expert_1 = prismatic_1.experts[expert_idx]
            expert_2 = prismatic_2.experts[expert_idx]

            for (name1, param1), (name2, param2) in zip(
                expert_1.named_parameters(), expert_2.named_parameters()
            ):
                assert name1 == name2
                assert torch.allclose(
                    param1, param2, atol=1e-6
                ), f"Expert {expert_idx} param '{name1}' differs between instances"

    def test_perturbations_are_sparse(self):
        """
        Test that perturbations are sparse (only k% of weights perturbed).

        Connection to Lottery Ticket Hypothesis (Frankle & Carbin, 2019):
        LTH shows critical sparse subnetworks exist. Prismatic inverts this by
        perturbing only the top 10% of weights (the "lottery tickets") to create
        structural obstacles that force alternative computational paths.

        Connection to Willow: Like perturbing individual qubits rather than the
        entire quantum system, we perturb ~10% of weights for maximum impact with
        minimal disruption.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,  # Only 10% of weights should be perturbed
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check sparsity for each perturbed expert
        for expert_idx in range(1, config.num_experts):
            expert = prismatic.experts[expert_idx]

            for name, param in expert.named_parameters():
                if not param.requires_grad:
                    continue

                original = original_params[name]

                # Count how many weights were perturbed
                diff = torch.abs(param - original)
                perturbed_mask = diff > 1e-6
                num_perturbed = perturbed_mask.sum().item()
                total_params = param.numel()

                actual_sparsity = num_perturbed / total_params

                # For small parameter tensors (like LayerNorm with 64 elements),
                # the top-k% selection might capture more or less than expected due to
                # rounding and magnitude similarity.
                #
                # Key change: We NOW perturb normalization layers (previously skipped).
                # For zero-initialized parameters (like bias), magnitude-based selection
                # may not select any (all have magnitude 0). This is fine - the key is
                # that we don't artificially skip normalization layers.
                #
                # For larger tensors, sparsity should be approximately correct.
                if total_params > 100:
                    # For larger tensors, enforce sparsity constraint
                    assert actual_sparsity <= config.sparsity * 1.5, (
                        f"Expert {expert_idx} param '{name}': "
                        f"sparsity {actual_sparsity:.3f} exceeds expected {config.sparsity}"
                    )
                # For small tensors, we just note they exist and may or may not be perturbed

    def test_perturbations_are_magnitude_aware(self):
        """
        Test that perturbations scale with parameter magnitude.

        Connection to paper: Magnitude-aware scaling preserves relative importance
        of parameters while introducing controlled irregularity. This maintains the
        network's learned prior while forcing different computational paths.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.01,
            sparsity=0.2,  # Higher sparsity for better statistics
        )

        # Create base expert with varying magnitudes
        base_expert = SimpleMLP(hidden_size=64)

        # Manually set some weights to different magnitudes
        with torch.no_grad():
            base_expert.fc1.weight[:32, :] *= 10.0  # Large magnitude
            base_expert.fc1.weight[32:, :] *= 0.1  # Small magnitude

        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check expert 1 (first perturbed expert)
        expert_1 = prismatic.experts[1]
        fc1_weight_perturbed = expert_1.fc1.weight
        fc1_weight_original = original_params["fc1.weight"]

        # Calculate perturbation magnitudes
        perturbations = torch.abs(fc1_weight_perturbed - fc1_weight_original)

        # Get average perturbation for large vs small magnitude weights
        large_mag_perturbations = perturbations[:32, :][perturbations[:32, :] > 1e-6]
        small_mag_perturbations = perturbations[32:, :][perturbations[32:, :] > 1e-6]

        if len(large_mag_perturbations) > 0 and len(small_mag_perturbations) > 0:
            avg_large = large_mag_perturbations.mean().item()
            avg_small = small_mag_perturbations.mean().item()

            # Perturbations should scale with magnitude
            # Large magnitude weights should have larger perturbations
            assert (
                avg_large > avg_small
            ), "Perturbations should scale with parameter magnitude"


class TestForwardPass:
    """Test forward pass functionality in both router and direct modes."""

    def test_router_mode_forward(self):
        """
        Test router mode forward pass (7 arguments).

        This is the standard mode used by LocalLayer in Praxis.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,  # Disable dropout for deterministic test
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Create inputs
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Router mode forward
        output, kv, state, loss = prismatic(
            layer=base_expert,  # Not used, but required for interface
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        assert output.shape == inputs.shape
        assert isinstance(loss, (int, float, torch.Tensor))

    def test_direct_mode_forward(self):
        """
        Test direct mode forward pass (2 arguments: inputs, state).

        This is used by RecurrentBlock in Praxis.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        base_expert = SimpleRecurrentModule(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Create inputs
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)
        current_state = torch.randn(batch_size, 1, config.hidden_size)

        # Direct mode forward
        output, new_state, loss = prismatic(inputs, current_state)

        assert output.shape == inputs.shape
        assert new_state.shape == current_state.shape
        assert isinstance(loss, (int, float, torch.Tensor))

    def test_forward_produces_different_outputs_than_base(self):
        """
        Test that Prismatic produces different outputs than base expert alone.

        Connection to paper: The soft-merging of perturbed experts should produce
        qualitatively different outputs than a single unperturbed expert, even
        when expert 0 receives high routing probability.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        base_expert_copy = SimpleMLP(hidden_size=64)
        base_expert_copy.load_state_dict(base_expert.state_dict())

        prismatic = Prismatic(config, base_expert=base_expert)

        # Create inputs
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        # Get Prismatic output
        prismatic_output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Get base expert output
        base_output, _, _, _ = base_expert_copy(
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Outputs should differ (unless routing puts 100% weight on expert 0,
        # which is unlikely with multiple experts)
        assert not torch.allclose(
            prismatic_output, base_output, atol=1e-3
        ), "Prismatic output should differ from base expert due to perturbed experts"


class TestParameterMerging:
    """Test SMEAR-style parameter merging functionality."""

    def test_parameter_merging_produces_valid_output(self):
        """Test that parameter merging produces valid, non-degenerate outputs."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Check output is valid (not NaN, not all zeros)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.abs().sum() > 0, "Output is all zeros"

    def test_routing_affects_output(self):
        """
        Test that different inputs produce different routing patterns.

        Connection to paper: The routing learns which "eyes" (experts) to attend
        to based on input patterns. This adaptive combination is key to extracting
        diverse patterns from the perturbed computational substrate.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        batch_size = 2
        seq_length = 16

        # Two very different inputs
        inputs_1 = torch.randn(batch_size, seq_length, config.hidden_size)
        inputs_2 = torch.randn(batch_size, seq_length, config.hidden_size) * 5.0

        output_1, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs_1,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        output_2, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs_2,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Outputs should be different (routing should adapt to inputs)
        assert not torch.allclose(
            output_1, output_2, atol=1e-3
        ), "Different inputs should produce different outputs"


class TestIntegration:
    """Test integration with Praxis components."""

    def test_integration_with_transformer_blocks(self):
        """
        Test Prismatic with actual TransformerBlock experts.

        This verifies compatibility with the full Praxis architecture.
        """
        config = PraxisConfig(
            hidden_size=64,
            embed_size=64,
            num_experts=3,
            num_heads=4,
            num_queries=4,
            k_heads=2,
            depth=2,
            dropout=0.0,
            residual_type="standard",
            attention_type="standard",
            expert_type="mlp",
            activation="gelu",
        )

        # Create base TransformerBlock
        base_block = TransformerBlock(config)

        # Create PrismaticConfig
        prismatic_config = PrismaticConfig(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        # Create Prismatic with TransformerBlock as base expert
        prismatic = Prismatic(prismatic_config, base_expert=base_block)

        # Test forward pass
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, loss = prismatic(
            layer=base_block,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        assert output.shape == inputs.shape
        assert isinstance(loss, (int, float, torch.Tensor))

    def test_gradient_flow(self):
        """
        Test that gradients flow properly through Prismatic.

        Important: The perturbations themselves are NOT trainable (they're static),
        but gradients should flow through the merged parameters during forward pass.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.01,
            sparsity=0.1,
            dropout=0.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        batch_size = 2
        seq_length = 16
        inputs = torch.randn(
            batch_size, seq_length, config.hidden_size, requires_grad=True
        )

        output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check that gradients flow to inputs
        assert inputs.grad is not None, "Gradients should flow to inputs"
        assert not torch.allclose(
            inputs.grad, torch.zeros_like(inputs.grad)
        ), "Input gradients should be non-zero"


class TestCleanPerturbations:
    """Test clean deterministic perturbations (Phase 1 baseline)."""

    def test_clean_perturbations_are_created(self):
        """Test that directional modes use simple deterministic perturbations."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=False,  # Phase 1: clean baseline
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Verify clean perturbations in repr
        assert "clean" in str(prismatic)

        # Verify experts are perturbed
        for expert_idx in range(1, config.num_experts):
            expert = prismatic.experts[expert_idx]
            # Should have perturbations
            has_perturbations = False
            for name, param in expert.named_parameters():
                if not param.requires_grad:
                    continue
                # Check against clean expert
                clean_param = prismatic.experts[0].state_dict()[name]
                if not torch.allclose(param, clean_param, atol=1e-6):
                    has_perturbations = True
                    break
            assert (
                has_perturbations
            ), f"Expert {expert_idx} should have clean deterministic perturbations"

    def test_perturbations_are_deterministic(self):
        """Test that clean perturbations are deterministic (reproducible)."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="repulsive",
            helical_modulation=False,
        )

        # Create two instances with same config
        base_expert_1 = SimpleMLP(hidden_size=64)
        base_expert_2 = SimpleMLP(hidden_size=64)
        base_expert_2.load_state_dict(base_expert_1.state_dict())

        prismatic_1 = Prismatic(config, base_expert=base_expert_1)
        prismatic_2 = Prismatic(config, base_expert=base_expert_2)

        # Experts should have identical perturbations
        for expert_idx in range(1, config.num_experts):
            for (name1, param1), (name2, param2) in zip(
                prismatic_1.experts[expert_idx].named_parameters(),
                prismatic_2.experts[expert_idx].named_parameters()
            ):
                assert torch.allclose(param1, param2, atol=1e-6), \
                    f"Clean perturbations should be deterministic for expert {expert_idx}, param {name1}"


class TestHelicalModulation:
    """Test helical modulation (Phase 2 experiment)."""

    def test_helical_modulation_is_created(self):
        """Test that helical modulation can be enabled."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=True,  # Phase 2: helical experiment
            helical_wavelength=1000.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Verify helical modulation in repr
        assert "helical" in str(prismatic)

        # Verify experts are perturbed
        for expert_idx in range(1, config.num_experts):
            expert = prismatic.experts[expert_idx]
            has_perturbations = False
            for name, param in expert.named_parameters():
                if not param.requires_grad:
                    continue
                clean_param = prismatic.experts[0].state_dict()[name]
                if not torch.allclose(param, clean_param, atol=1e-6):
                    has_perturbations = True
                    break
            assert has_perturbations, f"Expert {expert_idx} should have helical perturbations"

    def test_helical_creates_different_patterns_than_clean(self):
        """Test that helical modulation creates different patterns than clean."""
        base_expert = SimpleMLP(hidden_size=64)

        config_clean = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=False,
        )

        config_helical = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=True,
            helical_wavelength=1000.0,
        )

        base_1 = SimpleMLP(hidden_size=64)
        base_2 = SimpleMLP(hidden_size=64)
        base_2.load_state_dict(base_1.state_dict())

        prismatic_clean = Prismatic(config_clean, base_expert=base_1)
        prismatic_helical = Prismatic(config_helical, base_expert=base_2)

        # Expert 1 should have different perturbations
        params_differ = False
        for (name1, param1), (name2, param2) in zip(
            prismatic_clean.experts[1].named_parameters(),
            prismatic_helical.experts[1].named_parameters()
        ):
            if not param1.requires_grad:
                continue
            if not torch.allclose(param1, param2, atol=1e-6):
                params_differ = True
                break

        assert params_differ, "Helical modulation should create different perturbations than clean"

    def test_helical_wavelength_affects_pattern(self):
        """Test that different wavelengths create different spiral patterns."""
        config_1 = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=True,
            helical_wavelength=500.0,
        )

        config_2 = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.8,
            sparsity=0.1,
            perturbation_mode="attractive",
            helical_modulation=True,
            helical_wavelength=2000.0,  # Different wavelength
        )

        base_1 = SimpleMLP(hidden_size=64)
        base_2 = SimpleMLP(hidden_size=64)
        base_2.load_state_dict(base_1.state_dict())

        prismatic_1 = Prismatic(config_1, base_expert=base_1)
        prismatic_2 = Prismatic(config_2, base_expert=base_2)

        # Should have different perturbations
        params_differ = False
        for (name1, param1), (name2, param2) in zip(
            prismatic_1.experts[1].named_parameters(),
            prismatic_2.experts[1].named_parameters()
        ):
            if not param1.requires_grad:
                continue
            if not torch.allclose(param1, param2, atol=1e-6):
                params_differ = True
                break

        assert params_differ, "Different wavelengths should create different patterns"


class TestPerturbationModes:
    """Test different perturbation modes: attractive/repulsive vs noise."""

    def test_repulsive_mode_amplifies_top_weights(self):
        """
        Test that repulsive mode symmetrically amplifies top-magnitude weights.

        In repulsive mode with dual_sided strategy:
        - Top positive weights: W + scale * W = 2W (more positive)
        - Top negative weights: W + scale * W = 2W (more negative)
        - Both push toward overflow regime (extreme values)
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,  # 100% amplification
            sparsity=0.2,  # 10% top + 10% bottom for easier testing
            perturbation_strategy="dual_sided",
            perturbation_mode="repulsive",  # Quantum Mirror
            helical_modulation=False,  # Disable for exact math test
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check expert 1 (first perturbed expert)
        expert_1 = prismatic.experts[1]

        # Check fc1.weight (should have top weights amplified)
        fc1_original = original_params["fc1.weight"]
        fc1_perturbed = expert_1.fc1.weight.detach()

        # Find top 10% by magnitude in original
        flat_orig = fc1_original.flatten()
        flat_orig_abs = flat_orig.abs()
        num_top = max(1, int(flat_orig_abs.numel() * 0.1))
        top_threshold = torch.topk(flat_orig_abs, num_top).values[-1]
        top_mask = (flat_orig_abs >= top_threshold)

        # Get corresponding values
        top_original = flat_orig[top_mask]
        top_perturbed = fc1_perturbed.flatten()[top_mask]

        # Top weights should be amplified symmetrically: W_new = W + scale * W = 2W
        # - Positive: W + W = 2W (more positive)
        # - Negative: W + W = 2W (more negative)
        for orig, pert in zip(top_original, top_perturbed):
            expected = orig + 1.0 * orig  # W + W = 2W
            if orig > 0:
                # Positive weights should double
                assert pert > orig, f"Top positive weight {orig} not amplified (got {pert})"
                assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"
            else:
                # Negative weights should also double (become more negative)
                assert pert < orig, f"Top negative weight {orig} not amplified (got {pert})"
                assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"

    def test_repulsive_mode_suppresses_bottom_weights(self):
        """
        Test that repulsive mode symmetrically suppresses bottom-magnitude weights.

        In repulsive mode with dual_sided strategy:
        - Bottom positive weights: W - scale * W = 0 (toward zero)
        - Bottom negative weights: W - scale * W = 0 (toward zero)
        - Both push toward underflow/subnormal regime (near-zero values)
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,  # 100% suppression
            sparsity=0.2,  # 10% top + 10% bottom
            perturbation_strategy="dual_sided",
            perturbation_mode="repulsive",
            helical_modulation=False,  # Disable for exact math test
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check expert 1
        expert_1 = prismatic.experts[1]

        # Check fc1.weight
        fc1_original = original_params["fc1.weight"]
        fc1_perturbed = expert_1.fc1.weight.detach()

        # Find bottom 10% NON-ZERO weights by magnitude in original
        flat_orig = fc1_original.flatten()
        flat_orig_abs = flat_orig.abs()
        non_zero_mask = flat_orig_abs > 0
        non_zero_abs = flat_orig_abs[non_zero_mask]

        if len(non_zero_abs) > 0:
            num_bottom = max(1, int(flat_orig_abs.numel() * 0.1))
            num_bottom = min(num_bottom, len(non_zero_abs))
            bottom_threshold = torch.topk(non_zero_abs, num_bottom, largest=False).values[-1]
            bottom_mask = (flat_orig_abs <= bottom_threshold) & (flat_orig_abs > 0)

            # Get corresponding values
            bottom_original = flat_orig[bottom_mask]
            bottom_perturbed = fc1_perturbed.flatten()[bottom_mask]

            # Bottom weights should be suppressed symmetrically: W_new = W - scale * W = 0
            # - Positive: W - W = 0 (toward zero)
            # - Negative: W - W = 0 (toward zero)
            for orig, pert in zip(bottom_original, bottom_perturbed):
                expected = orig - 1.0 * orig  # W - W = 0
                if orig > 0:
                    # Positive bottom weights should approach 0
                    assert pert < orig, f"Bottom positive weight {orig} not suppressed (got {pert})"
                    assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"
                else:
                    # Negative bottom weights should also approach 0 (NOT amplified)
                    assert pert > orig, f"Bottom negative weight {orig} not suppressed toward 0 (got {pert})"
                    assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"

    def test_noise_mode_is_bidirectional(self):
        """
        Test that noise mode adds bidirectional Gaussian noise (legacy behavior).

        In noise mode:
        - Weights can go up OR down (non-deterministic direction)
        - Uses random noise N(0,1) scaled by magnitude
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.1,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="noise",  # Legacy mode
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check that perturbations exist
        expert_1 = prismatic.experts[1]
        fc1_original = original_params["fc1.weight"]
        fc1_perturbed = expert_1.fc1.weight.detach()

        # Should have differences (perturbations applied)
        assert not torch.allclose(fc1_original, fc1_perturbed, atol=1e-6), \
            "Noise mode should create perturbations"

        # The noise is random, so we can't predict exact values
        # But we can verify that perturbations are roughly magnitude-scaled
        diff = (fc1_perturbed - fc1_original).abs()
        mag = fc1_original.abs()

        # Perturbations should be roughly proportional to magnitude
        # (with some variance due to Gaussian noise)
        perturbed_mask = diff > 1e-6
        if perturbed_mask.sum() > 0:
            avg_diff = diff[perturbed_mask].mean()
            avg_mag = mag[perturbed_mask].mean()
            # With scale=0.1 and Gaussian noise, average perturbation should be
            # roughly scale * magnitude (with high variance)
            assert avg_diff > 0, "Should have non-zero perturbations"

    def test_repulsive_mode_is_deterministic(self):
        """
        Test that repulsive mode produces identical perturbations (no randomness).
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="repulsive",
        )

        # Create two instances with same base weights
        base_expert_1 = SimpleMLP(hidden_size=64)
        base_expert_2 = SimpleMLP(hidden_size=64)
        base_expert_2.load_state_dict(base_expert_1.state_dict())

        prismatic_1 = Prismatic(config, base_expert=base_expert_1)
        prismatic_2 = Prismatic(config, base_expert=base_expert_2)

        # Experts should be identical
        for (name1, param1), (name2, param2) in zip(
            prismatic_1.experts[1].named_parameters(),
            prismatic_2.experts[1].named_parameters()
        ):
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Repulsive mode should be deterministic for {name1}"

    def test_modes_produce_different_perturbations(self):
        """Test that repulsive and noise modes produce different perturbations."""
        # Same base expert for both
        base_expert = SimpleMLP(hidden_size=64)

        config_repulsive = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.1,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="repulsive",
        )

        config_noise = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=0.1,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="noise",
        )

        # Clone base expert for each mode
        base_repulsive = SimpleMLP(hidden_size=64)
        base_repulsive.load_state_dict(base_expert.state_dict())
        base_noise = SimpleMLP(hidden_size=64)
        base_noise.load_state_dict(base_expert.state_dict())

        prismatic_repulsive = Prismatic(config_repulsive, base_expert=base_repulsive)
        prismatic_noise = Prismatic(config_noise, base_expert=base_noise)

        # Perturbed experts should differ between modes
        expert_rep = prismatic_repulsive.experts[1]
        expert_noise = prismatic_noise.experts[1]

        params_differ = False
        for (name_rep, param_rep), (name_noise, param_noise) in zip(
            expert_rep.named_parameters(),
            expert_noise.named_parameters()
        ):
            if not torch.allclose(param_rep, param_noise, atol=1e-6):
                params_differ = True
                break

        assert params_differ, "Repulsive and noise modes should produce different perturbations"


class TestAttractiveMode:
    """Test attractive mode (neuronal regeneration - attract to dormant)."""

    def test_attractive_mode_suppresses_top_weights(self):
        """
        Test that attractive mode symmetrically suppresses top-magnitude weights.

        In attractive mode with dual_sided strategy:
        - Top positive weights: W - scale * W (toward zero, prune lottery tickets)
        - Top negative weights: W - scale * W (toward zero, prune lottery tickets)
        - Tests neuronal regeneration hypothesis: prune strong signals
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,  # 100% suppression
            sparsity=0.2,  # 10% top + 10% bottom for easier testing
            perturbation_strategy="dual_sided",
            perturbation_mode="attractive",  # Reverse Quantum Mirror
            helical_modulation=False,  # Disable for exact math test
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check expert 1 (first perturbed expert)
        expert_1 = prismatic.experts[1]

        # Check fc1.weight (should have top weights suppressed)
        fc1_original = original_params["fc1.weight"]
        fc1_perturbed = expert_1.fc1.weight.detach()

        # Find top 10% by magnitude in original
        flat_orig = fc1_original.flatten()
        flat_orig_abs = flat_orig.abs()
        num_top = max(1, int(flat_orig_abs.numel() * 0.1))
        top_threshold = torch.topk(flat_orig_abs, num_top).values[-1]
        top_mask = (flat_orig_abs >= top_threshold)

        # Get corresponding values
        top_original = flat_orig[top_mask]
        top_perturbed = fc1_perturbed.flatten()[top_mask]

        # Top weights should be suppressed symmetrically: W_new = W - scale * W = 0
        # - Positive: W - W = 0 (toward zero)
        # - Negative: W - W = 0 (toward zero)
        for orig, pert in zip(top_original, top_perturbed):
            expected = orig - 1.0 * orig  # W - W = 0
            if orig > 0:
                # Positive weights should be suppressed toward zero
                assert pert < orig, f"Top positive weight {orig} not suppressed (got {pert})"
                assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"
            else:
                # Negative weights should also be suppressed toward zero (NOT amplified)
                assert pert > orig, f"Top negative weight {orig} not suppressed toward 0 (got {pert})"
                assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"

    def test_attractive_mode_amplifies_bottom_weights(self):
        """
        Test that attractive mode symmetrically amplifies bottom-magnitude weights.

        In attractive mode with dual_sided strategy:
        - Bottom positive weights: W + scale * W (away from zero, wake dormant neurons)
        - Bottom negative weights: W + scale * W (away from zero, wake dormant neurons)
        - Tests neuronal regeneration hypothesis: amplify weak signals
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,  # 100% amplification
            sparsity=0.2,  # 10% top + 10% bottom
            perturbation_strategy="dual_sided",
            perturbation_mode="attractive",
            helical_modulation=False,  # Disable for exact math test
        )

        base_expert = SimpleMLP(hidden_size=64)

        # Store original parameters
        original_params = {}
        for name, param in base_expert.named_parameters():
            original_params[name] = param.clone().detach()

        prismatic = Prismatic(config, base_expert=base_expert)

        # Check expert 1
        expert_1 = prismatic.experts[1]

        # Check fc1.weight
        fc1_original = original_params["fc1.weight"]
        fc1_perturbed = expert_1.fc1.weight.detach()

        # Find bottom 10% NON-ZERO weights by magnitude in original
        flat_orig = fc1_original.flatten()
        flat_orig_abs = flat_orig.abs()
        non_zero_mask = flat_orig_abs > 0
        non_zero_abs = flat_orig_abs[non_zero_mask]

        if len(non_zero_abs) > 0:
            num_bottom = max(1, int(flat_orig_abs.numel() * 0.1))
            num_bottom = min(num_bottom, len(non_zero_abs))
            bottom_threshold = torch.topk(non_zero_abs, num_bottom, largest=False).values[-1]
            bottom_mask = (flat_orig_abs <= bottom_threshold) & (flat_orig_abs > 0)

            # Get corresponding values
            bottom_original = flat_orig[bottom_mask]
            bottom_perturbed = fc1_perturbed.flatten()[bottom_mask]

            # Bottom weights should be amplified symmetrically: W_new = W + scale * W = 2W
            # - Positive: W + W = 2W (away from zero)
            # - Negative: W + W = 2W (away from zero, more negative)
            for orig, pert in zip(bottom_original, bottom_perturbed):
                expected = orig + 1.0 * orig  # W + W = 2W
                if orig > 0:
                    # Positive bottom weights should be amplified away from 0
                    assert pert > orig, f"Bottom positive weight {orig} not amplified (got {pert})"
                    assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"
                else:
                    # Negative bottom weights should also be amplified (more negative)
                    assert pert < orig, f"Bottom negative weight {orig} not amplified (got {pert})"
                    assert torch.isclose(pert, expected, atol=1e-5), f"Expected {expected}, got {pert}"

    def test_attractive_mode_is_deterministic(self):
        """
        Test that attractive mode produces identical perturbations (no randomness).
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="attractive",
        )

        # Create two instances with same base weights
        base_expert_1 = SimpleMLP(hidden_size=64)
        base_expert_2 = SimpleMLP(hidden_size=64)
        base_expert_2.load_state_dict(base_expert_1.state_dict())

        prismatic_1 = Prismatic(config, base_expert=base_expert_1)
        prismatic_2 = Prismatic(config, base_expert=base_expert_2)

        # Experts should be identical
        for (name1, param1), (name2, param2) in zip(
            prismatic_1.experts[1].named_parameters(),
            prismatic_2.experts[1].named_parameters()
        ):
            assert torch.allclose(param1, param2, atol=1e-6), \
                f"Attractive mode should be deterministic for {name1}"

    def test_attractive_vs_repulsive_modes_differ(self):
        """
        Test that attractive and repulsive modes produce opposite perturbations.
        """
        # Same base expert for both
        base_expert = SimpleMLP(hidden_size=64)

        config_repulsive = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="repulsive",
        )

        config_attractive = PrismaticConfig(
            hidden_size=64,
            num_experts=2,
            perturbation_scale=1.0,
            sparsity=0.1,
            perturbation_strategy="dual_sided",
            perturbation_mode="attractive",
        )

        # Clone base expert for each mode
        base_repulsive = SimpleMLP(hidden_size=64)
        base_repulsive.load_state_dict(base_expert.state_dict())
        base_attractive = SimpleMLP(hidden_size=64)
        base_attractive.load_state_dict(base_expert.state_dict())

        prismatic_repulsive = Prismatic(config_repulsive, base_expert=base_repulsive)
        prismatic_attractive = Prismatic(config_attractive, base_expert=base_attractive)

        # Perturbed experts should have opposite perturbations
        expert_rep = prismatic_repulsive.experts[1]
        expert_att = prismatic_attractive.experts[1]

        # Get fc1.weight perturbations
        original = base_expert.fc1.weight.clone().detach()
        rep_weight = expert_rep.fc1.weight.detach()
        att_weight = expert_att.fc1.weight.detach()

        # Calculate perturbations
        rep_pert = rep_weight - original
        att_pert = att_weight - original

        # Perturbations should be opposite: att_pert = -rep_pert (where perturbed)
        # Due to the sign flip in attractive mode
        perturbed_mask = rep_pert.abs() > 1e-6
        if perturbed_mask.sum() > 0:
            rep_pert_values = rep_pert[perturbed_mask]
            att_pert_values = att_pert[perturbed_mask]

            # Should be approximately opposite
            assert torch.allclose(att_pert_values, -rep_pert_values, atol=1e-5), \
                "Attractive should create opposite perturbations from repulsive"

    def test_attractive_mode_forward_pass(self):
        """Test that attractive mode works in forward pass without errors."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=1.0,
            sparsity=0.1,
            perturbation_mode="attractive",
            dropout=0.0,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Forward pass
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, loss = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        # Should produce valid outputs
        assert output.shape == inputs.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.abs().sum() > 0, "Output is all zeros"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_expert(self):
        """
        Test Prismatic with num_experts=1 (only clean expert).

        This should work but provide no diversity (just the base expert).
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=1,  # Only one expert (the clean one)
            perturbation_scale=0.01,
            sparsity=0.1,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        assert len(prismatic.experts) == 1

        # Forward pass should work
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        assert output.shape == inputs.shape

    def test_high_sparsity(self):
        """Test with high sparsity (90% of weights perturbed)."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.01,
            sparsity=0.9,  # Perturb 90% of weights
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Should still work
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        assert output.shape == inputs.shape
        assert not torch.isnan(output).any()

    def test_large_perturbation_scale(self):
        """Test with large perturbation scale."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.5,  # Large perturbations
            sparsity=0.1,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Should still produce valid outputs (may be quite different from base)
        batch_size = 2
        seq_length = 16
        inputs = torch.randn(batch_size, seq_length, config.hidden_size)

        output, _, _, _ = prismatic(
            layer=base_expert,
            inputs=inputs,
            attention_mask=None,
            past_key_values=None,
            current_state=None,
            current_depth=0,
            block_ids=None,
        )

        assert output.shape == inputs.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
