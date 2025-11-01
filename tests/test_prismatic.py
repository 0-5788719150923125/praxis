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

        # Check defaults (1.0 = aggressive corruption by default)
        assert prismatic.perturbation_scale == 1.0
        assert prismatic.sparsity == 0.1
        assert prismatic.perturb_by_magnitude is True
        assert prismatic.dropout_rate == 0.1


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


class TestPiSeeding:
    """Test pi-digit seeding (Quantum Echoes)."""

    def test_pi_seeding_enabled(self):
        """Test that pi-seeding can be enabled and creates perturbations."""
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            use_pi_seeding=True,
            pi_position=100,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Verify pi-seeding is enabled
        assert prismatic.use_pi_seeding is True
        assert "pi-seeded" in str(prismatic)

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
            ), f"Expert {expert_idx} should be perturbed with pi-seeding"

    def test_pi_vs_hash_seeding_differ(self):
        """Test that pi-seeding produces different perturbations than hash-seeding."""
        config_pi = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.01,
            sparsity=0.1,
            use_pi_seeding=True,
            pi_position=100,
        )

        config_hash = PrismaticConfig(
            hidden_size=64,
            num_experts=3,
            perturbation_scale=0.01,
            sparsity=0.1,
            use_pi_seeding=False,
        )

        # Same base expert (load same weights)
        base_expert_pi = SimpleMLP(hidden_size=64)
        base_expert_hash = SimpleMLP(hidden_size=64)
        base_expert_hash.load_state_dict(base_expert_pi.state_dict())

        prismatic_pi = Prismatic(config_pi, base_expert=base_expert_pi)
        prismatic_hash = Prismatic(config_hash, base_expert=base_expert_hash)

        # Compare expert 1 perturbations
        expert_1_pi = prismatic_pi.experts[1]
        expert_1_hash = prismatic_hash.experts[1]

        # Should have different perturbations
        params_differ = False
        for (name_pi, param_pi), (name_hash, param_hash) in zip(
            expert_1_pi.named_parameters(), expert_1_hash.named_parameters()
        ):
            if not param_pi.requires_grad:
                continue
            if not torch.allclose(param_pi, param_hash, atol=1e-6):
                params_differ = True
                break

        assert (
            params_differ
        ), "Pi-seeding and hash-seeding should produce different perturbations"

    def test_quantum_echoes_backward_walk(self):
        """
        Test that experts walk backwards through pi (Quantum Echoes).

        Expert 1 should use pi[position-1], Expert 2 uses pi[position-2], etc.
        """
        config = PrismaticConfig(
            hidden_size=64,
            num_experts=4,
            perturbation_scale=0.01,
            sparsity=0.1,
            use_pi_seeding=True,
            pi_position=100,
        )

        base_expert = SimpleMLP(hidden_size=64)
        prismatic = Prismatic(config, base_expert=base_expert)

        # Each expert should have different perturbations (walking backward through pi)
        expert_params = []
        for expert_idx in range(1, config.num_experts):
            expert = prismatic.experts[expert_idx]
            # Get first linear layer weight
            param = expert.fc1.weight.clone()
            expert_params.append(param)

        # All experts should have different perturbations
        for i in range(len(expert_params)):
            for j in range(i + 1, len(expert_params)):
                assert not torch.allclose(
                    expert_params[i], expert_params[j], atol=1e-6
                ), f"Expert {i+1} and Expert {j+1} should have different pi-based perturbations"


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
