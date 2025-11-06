"""
Prismatic Attention: Multi-Eye Architectural Diversity Through Static Perturbations

This module implements prismatic attention as described in "The Blind Watchmaker" paper,
extending SMEAR routing with deterministic sparse perturbations to create architectural
diversity from a single base expert.

Theoretical Foundation:
---------------------
The paper argues that attention *constructs* which patterns manifest from the computational
substrate (floating-point approximation space), rather than merely filtering pre-existing
patterns. Different architectural constraints force different gradient trajectories through
this substrate, revealing patterns that single-architecture approaches cannot learn during
finite training.

Prismatic attention implements this by:
1. Cloning a base expert into N copies
2. Keeping one "clean" expert (Expert 0) - the baseline reality
3. Applying static, deterministic sparse perturbations to others - forced alternative realities
4. Using soft-merging (SMEAR) to adaptively combine these diverse traversals

Connection to Lottery Ticket Hypothesis:
---------------------------------------
The Lottery Ticket Hypothesis (Frankle & Carbin, 2019) states that dense neural networks
contain sparse subnetworks ("winning tickets") that can train to similar accuracy when
isolated. Prismatic inverts this: instead of finding critical sparse subnetworks through
pruning, we create sparse perturbations to force discovery of alternative computational paths.

By perturbing only 10% of weights (the "lottery tickets" of highest magnitude), we target
the most critical parameters while leaving 90% of the network intact. This creates structural
obstacles that force gradient descent to explore genuinely different regions of the loss
landscape, testing whether sparse architectural diversity reveals patterns that uniform
approaches cannot discover.

Connection to Quantum Computing (Willow Architecture & Quantum Echoes):
----------------------------------------------------------------------
Google's Willow quantum computer achieves error correction by perturbing individual qubits
in controlled ways. The key insight: perturbing 1 qubit (one unit of quantum information)
is sufficient to reveal error-correcting structure across the entire quantum system.

Prismatic implements "Quantum Echoes" - each expert is a temporal lag, corrupted by different
artifacts. Like quantum decoherence creating echoes of the original state, perturbed experts
are echoes of the clean expert, each corrupted differently by mathematical noise.

In continuous neural networks, we cannot perturb "1 neuron" - but we can perturb a small
percentage (default: 10%) of the most important weights. Like Willow's single-qubit
perturbations, sparse weight perturbations create just enough irregularity to force the
network to discover robust, diverse computational paths without destroying functionality.

The perturbations are NOT learned - they are architectural constraints that force different
paths through the parameter space, analogous to how qubit perturbations force different
paths through quantum state space.

Helical Modulation (Structure Transfer Experiment):
----------------------------------------------------
Perturbations are modulated by helical/spiral patterns using Euler's formula:

    spiral(position, expert_idx) = cos(2π · position / wavelength + phase_offset)

Where:
- wavelength = π × 1000 (using π's actual mathematical value)
- phase_offset = expert_idx × 2π / num_experts (harmonic phase relationships)

Each expert experiences a different phase of the spiral pattern:
- Expert 0: Clean (unperturbed) - the baseline
- Expert 1: Phase offset π/N - first harmonic
- Expert 2: Phase offset 2π/N - second harmonic
- Expert 3: Phase offset 3π/N - third harmonic
- ...

During iterative reasoning (depth>layers), the model experiences different harmonic
combinations at each iteration. The soft-merging creates wave interference patterns
as experts combine.

**Hypothesis**: External helical structure in perturbations transfers into internal
patterns (gradients, learned features). This is testable by comparing helical_modulation
enabled vs disabled.

Key Insight: "Train on consensus, you manifest the lowest common denominator."
Static perturbations prevent convergence to consensus, maintaining genuine diversity throughout
training. Each expert traverses a fundamentally different computational substrate.
"""

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrismaticConfig:
    """Configuration for Prismatic attention module.

    Attributes:
        hidden_size: Hidden dimension size
        num_experts: Number of expert clones (including 1 clean expert)
        perturbation_scale: Scale factor for perturbations (default: 0.8)
            For inverse_directional (DEFAULT):
                0.5 = 50% pruning/amplification (conservative)
                0.8 = 80% pruning/amplification (default, balanced)
                0.9 = 90% pruning/amplification (aggressive)
                1.0 = 100% pruning (top→0), 100% amplification (bottom→2W)
                >1.0 = overshoots, flips signs (not recommended)
            For directional mode:
                1.0 = 100% amplification (top→2W, bottom→0) (standard)
                0.5 = 50% amplification (moderate)
            For noise mode:
                Scale controls Gaussian noise std = scale * |W|
        sparsity: Fraction of weights to perturb (default: 0.1 = 10%)
        perturb_by_magnitude: If True, perturb by magnitude; else random
        perturbation_strategy: Strategy for selecting weights to perturb
            - "dual_sided": Top sparsity/2 + bottom sparsity/2 by magnitude (default)
            - "top_only": Top sparsity by magnitude (original approach)
            - "bottom_only": Bottom sparsity by magnitude
            - "random": Random sparsity selection
        perturbation_mode: Mode for applying perturbations (default: "attractive")
            - "attractive": Neuronal regeneration - wake dormant, prune dominant (DEFAULT)
                Top weights: W - scale * W (suppress toward 0, prune lottery tickets)
                Bottom weights: W + scale * W (amplify away from 0, wake dormant neurons)
                Attracts attention to overlooked pathways with productive instability
            - "repulsive": Extreme exploration - push to numerical limits
                Top weights: W + scale * W (amplify to extremes, both + and -)
                Bottom weights: W - scale * W (suppress toward 0, both + and -)
                Repels weights to floating-point precision extremes (overflow/underflow)
            - "noise": Bidirectional Gaussian noise (legacy mode)
                Top weights: W + scale * |W| * N(0,1)
                Bottom weights: W + scale * |W| * N(0,1)
        dropout: Dropout probability for expert dropout during training
        focal_pattern: Pattern for modulating perturbations (default: "radial_helical")
            - "radial_helical": Radial lens + helical waves (DEFAULT - Prismatic lens)
                Each expert focuses at different radial positions with helical modulation
                Creates position-dependent transformation signatures with wave structure
                Focal points: π-based spacing, helical waves with π wavelength
            - "helical": Pure helical/spiral modulation using Euler's formula
                Perturbations modulated by cos(2π·position/wavelength + phase)
                Creates wave interference patterns when experts merge
            - "radial": Pure radial lens focusing
                Each expert focuses at different positions in weight space
                Creates hierarchical center-to-edge gradients in transformations
            - "none": No modulation - simple deterministic perturbations
                Clean baseline for ablation studies
        focal_length: Focal length for radial lens (default: π * 100)
            Controls how quickly focal strength decays from center
        helical_wavelength: Wavelength for helical pattern (default: π * 1000)
            Creates harmonic relationships between experts with phase offsets
    """

    hidden_size: int
    num_experts: int
    perturbation_scale: float = (
        0.8  # For attractive: top→0.2W (prune 80%), bottom→1.8W (amplify 80%)
    )
    sparsity: float = 0.1  # 10% total: 5% top + 5% bottom
    perturb_by_magnitude: bool = True
    perturbation_strategy: str = "dual_sided"
    perturbation_mode: str = "attractive"
    dropout: float = 0.0
    focal_pattern: str = "radial_helical"  # Lens + waves (default)
    focal_length: float = math.pi * 100  # π × 100 - Gaussian lens decay rate
    helical_wavelength: float = math.pi * 1000  # π × 1000 - harmonic wave period


class Prismatic(nn.Module):
    """
    Prismatic Attention: Creates architectural diversity through static perturbations.

    This module implements the "dual-stream" or "multi-eye" architecture described in
    "The Blind Watchmaker" paper. It creates N expert clones from a single base expert,
    where:

    - Expert 0 (the "right eye"): Clean, unperturbed - represents consensus reality
    - Experts 1..N (the "left eye(s)"): Statically perturbed - forced alternative realities

    The perturbations are:
    - Deterministic (pi-digit or hash-based seeding for reproducibility)
    - Sparse (only k% of weights, focusing on high/low-magnitude parameters)
    - Static (not trainable - architectural constraints, not optimization targets)
    - Magnitude-aware (scaled by existing weight magnitudes)

    Perturbation Modes:
    ------------------
    1. "attractive" mode (DEFAULT - Neuronal Regeneration):
       - Attracts attention to dormant pathways, prunes dominant signals
       - Top 5%: W - scale * W (suppress strong signals, prune lottery tickets)
         - Positive → toward 0 (disrupt learned structure)
         - Negative → toward 0 (disrupt learned structure)
       - Bottom 5%: W + scale * W (amplify weak signals, wake dormant neurons)
         - Positive → more positive (large relative perturbation)
         - Negative → more negative (large relative perturbation)
       - Creates productive instability through forced neuronal turnover
       - Tests whether continuous pruning/restoration cycles aid learning
       - Deterministic - no random noise

    2. "repulsive" mode (Extreme Polarization):
       - Repels weights to numerical extremes for precision regime exploration
       - Top 5%: W + scale * W (amplify to extremes, repel from center)
         - Positive → more positive (overflow regime)
         - Negative → more negative (overflow regime)
       - Bottom 5%: W - scale * W (suppress toward zero)
         - Positive → toward 0 (underflow/subnormal regime)
         - Negative → toward 0 (underflow/subnormal regime)
       - Explores ALL floating-point precision regimes symmetrically
       - Deterministic - no random noise

    3. "noise" mode (legacy):
       - Adds bidirectional Gaussian noise to selected weights
       - Top 5%: W + scale * |W| * N(0,1) (can go up or down)
       - Bottom 5%: W + scale * |W| * N(0,1) (can go up or down)
       - Creates architectural chaos at both numerical extremes

    Why Static Perturbations?
    ------------------------
    If perturbations were trainable, they would optimize toward similar solutions
    (convergence). Static perturbations maintain persistent architectural asymmetry,
    forcing each expert to adapt to an irregular parameter space it didn't choose.

    This tests the hypothesis: Can forced architectural diversity reveal patterns that
    single-architecture approaches cannot learn, regardless of scale?

    Connection to Paper's Core Thesis:
    ---------------------------------
    "Attention is a constructor, not a filter."

    If attention constructs which patterns manifest (rather than filtering pre-existing
    patterns), then architectural diversity must be inherent and persistent. Static
    perturbations ensure genuine diversity throughout training, enabling parallel
    exploration of different regions in the computational substrate.
    """

    __version__ = "0.1.0"

    def __init__(
        self, config: Any, layout: str = "standard", *args: Any, **kwargs: Any
    ):
        """
        Initialize Prismatic attention module.

        Args:
            config: Configuration object with num_experts, hidden_size, etc.
            layout: Layout type (kept for interface compatibility)
            *args: Additional arguments
            **kwargs: Must contain 'base_expert' - the module to clone and perturb
        """
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        self.perturbation_scale = getattr(config, "perturbation_scale", 0.8)
        self.sparsity = getattr(config, "sparsity", 0.1)
        self.perturb_by_magnitude = getattr(config, "perturb_by_magnitude", True)
        self.perturbation_strategy = getattr(
            config, "perturbation_strategy", "dual_sided"
        )
        self.perturbation_mode = getattr(config, "perturbation_mode", "attractive")
        self.dropout_rate = getattr(config, "dropout", 0.1)
        self.focal_pattern = getattr(config, "focal_pattern", "radial_helical")
        self.focal_length = getattr(config, "focal_length", math.pi * 100)
        self.helical_wavelength = getattr(config, "helical_wavelength", math.pi * 1000)

        # Get base expert from kwargs
        # Supports two initialization patterns:
        # 1. Direct: base_expert=<single_module> (for standalone usage)
        # 2. Praxis flow: experts=[list_of_modules] (standard LocalLayer pattern)
        base_expert = kwargs.get("base_expert", None)
        experts_list = kwargs.get("experts", None)

        if base_expert is None and experts_list is None:
            raise ValueError(
                "Prismatic router requires either 'base_expert' or 'experts' to be provided"
            )

        # If experts list is provided (standard Praxis flow), use first as base
        if base_expert is None and experts_list is not None:
            if not isinstance(experts_list, (list, tuple)) or len(experts_list) == 0:
                raise ValueError(
                    "When using 'experts' parameter, must provide non-empty list"
                )
            base_expert = experts_list[0]
            # Note: We ignore the other experts in the list since Prismatic
            # creates its own perturbed views at runtime from the base expert

        # Store the base expert (no cloning, no perturbation yet)
        # Perturbations will be applied at runtime during forward pass
        self.base_expert = base_expert

        # Router network with layer normalization (following SMEAR design)
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        # Track parameter names for merging
        self.parameter_names: List[str] = []

        # Metrics storage for convergence tracking
        self._metrics = {}

        # Dynamics metrics storage (gradient tracking)
        self._dynamics_metrics = {}

    def _create_perturbed_view(
        self, base_params: Dict[str, torch.Tensor], expert_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Create a perturbed view of base expert parameters at runtime.

        This applies perturbations to the current learned parameters before merging,
        ensuring architectural diversity persists as the base expert trains.

        Expert 0: Clean (returns base_params unmodified) - the "right eye" baseline
        Experts 1+: Perturbed views - the "left eye(s)" with forced irregularities

        Unlike the old approach (perturb once at init), this ensures:
        - As the base expert learns, perturbations are always relative to current state
        - Architectural diversity persists throughout training
        - No gradient convergence between "experts" (they're all views of one expert)

        Args:
            base_params: Base expert's current parameters
            expert_idx: Expert index (0 = clean, 1+ = perturbed)

        Returns:
            Dictionary of perturbed parameters (or clean if expert_idx=0)
        """
        if expert_idx == 0:
            # Expert 0: Return clean parameters
            return base_params

        # Expert 1+: Apply perturbations to current parameters
        perturbed_params = {}

        for param_name, param in base_params.items():
            # Create generator for reproducibility
            generator = torch.Generator(device=param.device).manual_seed(expert_idx)

            # Create sparse mask
            mask = self._create_sparse_mask(param, generator)

            # Generate perturbation
            perturbation = self._generate_perturbation(
                param, mask, generator, expert_idx, param_name
            )

            # Apply perturbation (non-destructive - creates new tensor)
            perturbed_params[param_name] = param + perturbation

        return perturbed_params


    def _create_sparse_mask(
        self, param: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """
        Create sparse mask indicating which parameters to perturb.

        Strategies:
        1. Dual-sided (DEFAULT): Perturb top sparsity/2 + bottom sparsity/2 by magnitude
           - Top weights: Expose coarse-grained float32 artifacts (large absolute rounding)
           - Bottom weights: Expose fine-grained float32 artifacts (large relative rounding)
           - Both extremes reveal different numerical precision regimes
           - Creates symmetry in exploration of the computational substrate
           - May activate dormant pathways suppressed during training

           Perturbation modes:
           - "noise": Both top and bottom get bidirectional noise
           - "repulsive": Top amplified (+), bottom suppressed (-)
           - "attractive": Top suppressed (-), bottom amplified (+)

        2. Top-only: Perturb top-k% by absolute magnitude (original approach)
           - Targets the "lottery tickets" - most critical parameters
           - Maximum impact through exponential cascade amplification
           - Aligns with standard Lottery Ticket Hypothesis interpretation

        3. Bottom-only: Perturb bottom-k% by absolute magnitude
           - Targets suppressed/dormant parameters
           - May wake up alternative computational paths
           - Tests if low-magnitude weights contain latent patterns

        4. Random: Randomly select k% of weights
           - Uniform distribution across parameter space
           - Control for ablation studies

        Default sparsity: 10% of weights
        - Dual-sided: 5% from top + 5% from bottom
        - Creates enough irregularity to force exploration while maintaining stability

        Args:
            param: Parameter tensor to create mask for
            generator: Seeded random generator for reproducibility

        Returns:
            Tiered mask for directional mode:
                +1.0 = Top tier (amplify in directional mode)
                -1.0 = Bottom tier (suppress in directional mode)
                0.0 = Middle tier (unchanged)
            For noise mode, absolute value is used (binary mask).
        """
        num_params = param.numel()

        if not self.perturb_by_magnitude or self.perturbation_strategy == "random":
            # Random selection strategy - binary mask (no tiers)
            num_perturbed = max(1, int(num_params * self.sparsity))
            mask = torch.zeros_like(param)
            flat_mask = mask.flatten()
            indices = torch.randperm(num_params, generator=generator)[:num_perturbed]
            flat_mask[indices] = 1.0
            mask = flat_mask.reshape(param.shape)
            return mask

        flat_param = param.flatten().abs()

        if self.perturbation_strategy == "dual_sided":
            # Dual-sided: Top half + Bottom half by magnitude
            # This exposes BOTH numerical precision regimes of float32
            num_per_side = max(1, int(num_params * self.sparsity / 2))

            # Top side: Highest magnitude weights
            # These are the "lottery tickets" - high impact through exponential cascade
            # For directional mode: these will be AMPLIFIED (+1)
            top_threshold = torch.topk(flat_param, num_per_side).values[-1]
            top_mask = (flat_param >= top_threshold).float()

            # Bottom side: Lowest magnitude NON-ZERO weights
            # These live in the fine-grained precision regime
            # Near activation thresholds - small perturbations can flip dead->alive
            # For directional mode: these will be SUPPRESSED (-1)
            non_zero_param = flat_param[flat_param > 0]
            if len(non_zero_param) >= num_per_side:
                # Use topk with largest=False to get smallest values
                bottom_threshold = torch.topk(
                    non_zero_param, num_per_side, largest=False
                ).values[-1]
                bottom_mask = (
                    (flat_param <= bottom_threshold) & (flat_param > 0)
                ).float()
            else:
                # Not enough non-zero weights, just use what we have
                bottom_mask = (flat_param > 0).float()

            # Create signed tiered mask:
            # +1 for top tier (amplify), -1 for bottom tier (suppress), 0 for middle
            mask = (top_mask - bottom_mask).reshape(param.shape)

        elif self.perturbation_strategy == "top_only":
            # Original approach: Top-k by magnitude only (positive mask)
            num_perturbed = max(1, int(num_params * self.sparsity))
            threshold = torch.topk(flat_param, num_perturbed).values[-1]
            mask = (flat_param >= threshold).float().reshape(param.shape)

        elif self.perturbation_strategy == "bottom_only":
            # Bottom-k by magnitude (positive mask for consistency)
            num_perturbed = max(1, int(num_params * self.sparsity))
            non_zero_param = flat_param[flat_param > 0]
            if len(non_zero_param) >= num_perturbed:
                threshold = torch.topk(
                    non_zero_param, num_perturbed, largest=False
                ).values[-1]
                mask = (
                    ((flat_param <= threshold) & (flat_param > 0))
                    .float()
                    .reshape(param.shape)
                )
            else:
                mask = (flat_param > 0).float().reshape(param.shape)
        else:
            raise ValueError(
                f"Unknown perturbation_strategy: {self.perturbation_strategy}. "
                f"Must be one of: dual_sided, top_only, bottom_only, random"
            )

        return mask

    def _generate_perturbation(
        self,
        param: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator,
        expert_idx: int,
        param_name: str,
    ) -> torch.Tensor:
        """
        Generate adaptive perturbation scaled by parameter magnitude.

        Three perturbation modes:
        1. "attractive": Suppress top, amplify bottom (neuronal regeneration)
        2. "repulsive": Amplify top, suppress bottom (extreme exploration)
        3. "noise": Bidirectional Gaussian noise (legacy)

        Four focal patterns for modulation:
        1. "radial_helical" (DEFAULT): Lens focusing + helical waves
           - Each expert focuses at different radial positions
           - Helical modulation creates wave structure
           - Creates transformation signatures: radial hierarchy + periodic oscillation

        2. "helical": Pure wave modulation (original)
           - Spiral patterns using Euler's formula
           - Phase offsets create harmonic relationships
           - Creates wave interference when experts merge

        3. "radial": Pure lens focusing
           - Each expert focuses at different positions in weight space
           - Creates center-to-edge gradients in transformations
           - Hierarchical structure only

        4. "none": No modulation
           - Simple deterministic perturbations
           - Clean baseline for ablation

        Args:
            param: Original parameter tensor
            mask: Tiered mask (+1 top, -1 bottom, 0 middle) or binary mask
            generator: Seeded random generator (used only for noise mode)
            expert_idx: Expert index for focal point and phase offset
            param_name: Parameter name (unused, kept for interface compatibility)

        Returns:
            Perturbation tensor (same shape as param)
        """
        import math

        if (
            self.perturbation_mode == "repulsive"
            or self.perturbation_mode == "attractive"
        ):
            # Calculate base perturbation (before modulation)
            if self.perturbation_mode == "attractive":
                # Suppress top, amplify bottom (attract to dormant)
                base_perturbation = -self.perturbation_scale * param * mask
            else:
                # Amplify top, suppress bottom (repel to extremes)
                base_perturbation = self.perturbation_scale * param * mask

            # Apply focal pattern modulation
            if expert_idx == 0 or self.focal_pattern == "none":
                # Expert 0 is always clean, or no modulation requested
                perturbation = base_perturbation
            else:
                # Calculate modulation based on focal pattern
                num_weights = param.numel()
                positions = torch.arange(
                    num_weights, dtype=param.dtype, device=param.device
                )

                if self.focal_pattern == "radial_helical":
                    # RADIAL-HELICAL: Prismatic lens (DEFAULT)
                    # Each expert focuses at different radial positions with helical waves

                    # Focal point varies by expert (distribute across weight space)
                    focal_point = (expert_idx / self.num_experts) * num_weights

                    # Radial distance from focal point
                    distance = torch.abs(positions - focal_point)

                    # Gaussian-like lens focusing using π-based focal length
                    # Stronger at focal point, decays with distance
                    focal_strength = torch.exp(-distance / self.focal_length)

                    # Helical modulation with expert-specific phase
                    positions_normalized = 2 * math.pi * distance / self.helical_wavelength
                    phase_offset = expert_idx * 2 * math.pi / self.num_experts
                    helical = torch.cos(positions_normalized + phase_offset)

                    # Combine: radial focusing × helical waves
                    # Creates spiral patterns radiating from focal point
                    modulation = focal_strength * helical
                    modulation = modulation.reshape(param.shape)
                    perturbation = base_perturbation * modulation

                elif self.focal_pattern == "radial":
                    # RADIAL: Pure lens focusing
                    # Each expert focuses at different positions in weight space

                    # Focal point varies by expert
                    focal_point = (expert_idx / self.num_experts) * num_weights

                    # Radial distance from focal point
                    distance = torch.abs(positions - focal_point)

                    # Gaussian-like lens focusing using π-based focal length
                    focal_strength = torch.exp(-distance / self.focal_length)
                    focal_strength = focal_strength.reshape(param.shape)

                    perturbation = base_perturbation * focal_strength

                elif self.focal_pattern == "helical":
                    # HELICAL: Pure wave modulation (original approach)
                    # Creates harmonic phase relationships between experts

                    # Normalize positions to wavelength
                    positions_normalized = 2 * math.pi * positions / self.helical_wavelength

                    # Expert-specific phase offset
                    phase_offset = expert_idx * 2 * math.pi / self.num_experts

                    # Create spiral pattern
                    spiral = torch.cos(positions_normalized + phase_offset)
                    spiral = spiral.reshape(param.shape)

                    perturbation = base_perturbation * spiral
                else:
                    raise ValueError(
                        f"Unknown focal_pattern: {self.focal_pattern}. "
                        f"Must be one of: radial_helical, helical, radial, none"
                    )
        else:
            # Noise mode: Bidirectional Gaussian noise
            # Scale by parameter magnitude (adaptive)
            magnitude_scale = param.abs()

            # Convert signed mask to binary (for dual_sided) or use as-is (for others)
            binary_mask = mask.abs()

            # Generate standard normal noise
            noise = torch.randn(
                param.shape, dtype=param.dtype, device=param.device, generator=generator
            )

            # Apply: scale * magnitude * noise * |mask|
            perturbation = (
                self.perturbation_scale * magnitude_scale * noise * binary_mask
            )

        return perturbation

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_experts={self.num_experts}, "
            f"strategy={self.perturbation_strategy}, "
            f"mode={self.perturbation_mode}, "
            f"scale={self.perturbation_scale}, "
            f"sparsity={self.sparsity}, "
            f"focal={self.focal_pattern})"
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], float],  # Direct mode
        Tuple[
            torch.Tensor,
            Optional[Union[torch.Tensor, List, Dict]],
            Optional[torch.Tensor],
            float,
        ],  # Router mode
    ]:
        """
        Forward pass with Prismatic routing.

        Implements SMEAR-style soft-merging: dynamically merge expert parameters
        based on routing probabilities, then apply the merged parameters using
        functional_call.

        This combines:
        1. Architectural diversity (static perturbations)
        2. Adaptive combination (learned routing)

        The routing learns which "eyes" (experts) to attend to, while the
        perturbations force each eye to see a fundamentally different reality.

        Supports two modes (following SMEAR):
        1. Direct mode: (inputs, current_state) -> used by RecurrentBlock
        2. Router mode: (layer, inputs, ...) -> used as router in LocalLayer

        Returns:
            Appropriate tuple based on mode
        """
        # Determine mode and parse arguments
        if self._is_router_mode(args, kwargs):
            router_args = self._parse_router_args(args, kwargs)
            return self._router_forward(*router_args)
        else:
            inputs, current_state = self._parse_direct_args(args, kwargs)
            return self._direct_forward(inputs, current_state)

    def _router_forward(
        self,
        layer: nn.Module,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Union[torch.Tensor, List, Dict]],
        current_state: Optional[torch.Tensor],
        current_depth: int,
        block_ids: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[Union[torch.Tensor, List, Dict]],
        Optional[torch.Tensor],
        float,
    ]:
        """
        Router mode forward pass.

        Soft-merges expert parameters based on routing probabilities, then
        applies merged parameters to process inputs.
        """
        # Compute routing probabilities with normalization
        router_input = inputs.mean(dim=1)  # [batch_size, hidden_size]
        router_input = self.router_norm(router_input)

        # Normalized router weights for stable training
        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Expert dropout during training (following SMEAR)
        if self.training and self.dropout_rate > 0:
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Soft-merge expert parameters based on routing probabilities
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        # Use base expert as structure for functional_call
        # Apply merged parameters (which are perturbed views soft-merged by routing)
        forward_args = (
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )

        result = torch.func.functional_call(
            self.base_expert, merged_state_dict, forward_args, {}
        )

        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 4:
            return result
        elif isinstance(result, tuple) and len(result) == 3:
            return result[0], result[1], result[2], 0.0
        else:
            return result, past_key_values, current_state, 0.0

    def _direct_forward(
        self,
        inputs: torch.Tensor,
        current_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Direct mode forward pass for RecurrentBlock usage.

        Simpler interface: (inputs, state) -> (outputs, state, aux_loss)
        """
        # Compute routing probabilities
        router_input = inputs.mean(dim=1)
        router_input = self.router_norm(router_input)

        normalized_weight = F.normalize(self.router.weight, dim=1)
        logits = F.linear(router_input, normalized_weight, self.router.bias)

        routing_probs = F.softmax(logits, dim=-1)

        # Expert dropout
        if self.training and self.dropout_rate > 0:
            expert_mask = torch.bernoulli(
                torch.ones_like(routing_probs) * (1 - self.dropout_rate)
            )
            routing_probs = routing_probs * expert_mask
            routing_probs = routing_probs / (
                routing_probs.sum(dim=-1, keepdim=True) + 1e-8
            )

        # Merge and apply
        merged_state_dict = self._merge_expert_parameters(routing_probs)

        result = torch.func.functional_call(
            self.base_expert, merged_state_dict, (inputs, current_state), {}
        )

        # Handle return formats
        if isinstance(result, tuple) and len(result) == 3:
            return result
        elif isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1], 0.0
        else:
            return result, current_state, 0.0

    def _merge_expert_parameters(
        self, routing_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Soft-merge expert parameters based on routing probabilities.

        This implements SMEAR's core mechanism with runtime perturbations:
        1. Get base expert's current parameters
        2. For each "expert view", apply perturbations to current params
        3. Merge the perturbed views based on routing probabilities

        Mathematical formulation:
            W_merged = Σ_i (p_i * W_perturbed_i)

        where p_i are routing probabilities and W_perturbed_i are perturbed views
        of the current base expert parameters.

        Connection to Prismatic Attention:
        ---------------------------------
        This merging adaptively combines different "eyes" (perturbed views).
        Perturbations are applied EVERY forward pass relative to current learned state,
        ensuring architectural diversity persists as the base expert trains.

        Args:
            routing_probs: Routing probabilities [batch_size, num_experts]

        Returns:
            Dictionary of merged parameters
        """
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # Mean routing probability across batch for each expert
        expert_weights = routing_probs.mean(dim=0)  # [num_experts]

        # Log expert convergence metrics for visualization
        self._log_routing_metrics(expert_weights, routing_probs)

        # Get base expert's current parameters
        base_state_dict = dict(self.base_expert.named_parameters())

        # Collect parameter names from base expert
        self.parameter_names = self._collect_parameter_names(self.base_expert)

        # Create perturbed views for each expert and merge
        for param_name in self.parameter_names:
            merged_param: Optional[torch.Tensor] = None

            for expert_idx in range(self.num_experts):
                # Get perturbed view of this parameter
                perturbed_view = self._create_perturbed_view(base_state_dict, expert_idx)
                param = perturbed_view[param_name]

                # Ensure device compatibility
                if param.device != expert_weights.device:
                    param = param.to(expert_weights.device)

                # Weight by routing probability
                weighted_param = param * expert_weights[expert_idx]

                if merged_param is None:
                    merged_param = weighted_param
                else:
                    merged_param = merged_param + weighted_param

            assert merged_param is not None
            merged_state_dict[param_name] = merged_param

        return merged_state_dict

    def _collect_parameter_names(
        self, module: nn.Module, prefix: str = ""
    ) -> List[str]:
        """Recursively collect all parameter names from module."""
        parameter_names = []
        for name, submodule in module.named_children():
            parameter_names.extend(
                self._collect_parameter_names(submodule, prefix + name + ".")
            )
        for name, _ in module.named_parameters(recurse=False):
            parameter_names.append(prefix + name)
        return parameter_names

    def _get_module_parameter(
        self, module: nn.Module, param_name: str
    ) -> Optional[torch.Tensor]:
        """Retrieve parameter from module using fully qualified name."""
        parts = param_name.split(".")
        submodule = module
        for part in parts[:-1]:
            if hasattr(submodule, part):
                submodule = getattr(submodule, part)
            else:
                return None
        return getattr(submodule, parts[-1], None)


    def _log_routing_metrics(
        self, expert_weights: torch.Tensor, routing_probs: torch.Tensor
    ) -> None:
        """
        Store routing metrics for expert convergence tracking.

        Tracks how routing probabilities evolve over training to visualize
        convergence patterns similar to Figure 1 in "The Blind Watchmaker" paper.

        For Prismatic attention:
        - Expert 0 (clean): The "right eye" - consensus reality
        - Experts 1+ (perturbed): The "left eye(s)" - forced alternative realities

        Metrics flow: Prismatic router → Decoder.get_metrics() → Model.get_metrics() →
                     BackpropagationTrainer.log_dict() → MetricsLoggerCallback →
                     SQLite → API → Web dashboard

        Args:
            expert_weights: Mean routing probability per expert [num_experts]
            routing_probs: Full routing probabilities [batch_size, num_experts]
        """
        try:
            # Per-expert routing weights (mean across batch)
            # Track convergence for each "eye" independently
            for i, weight in enumerate(expert_weights):
                self._metrics[f"expert_{i}_routing_weight"] = weight.item()

            # Entropy: H = -Σ(p_i * log(p_i))
            # Measures routing balance: high = balanced, low = collapsed
            probs = expert_weights + 1e-10  # Avoid log(0)
            entropy = -(probs * probs.log()).sum()
            self._metrics["routing_entropy"] = entropy.item()

            # Concentration: max routing weight
            # Tests if routing collapses to clean expert or maintains diversity
            concentration = expert_weights.max()
            self._metrics["routing_concentration"] = concentration.item()

            # Variance: measures routing stability across experts
            # High variance = specialized eyes, low variance = uniform
            variance = expert_weights.var()
            self._metrics["routing_variance"] = variance.item()
        except Exception:
            # Silently fail if metric computation fails - don't break training
            pass

    def log_gradient_dynamics(self) -> Optional[Dict]:
        """
        Log base expert gradient dynamics by weight tier.

        With the new runtime perturbation approach, there's only one base expert
        that receives gradients. We track gradient flow through different weight
        tiers (top/bottom/middle magnitude) to understand learning dynamics.

        Should be called after backward() but before optimizer.step().

        Returns:
            Dict with base expert gradients by tier, or None if no gradients available
        """
        if not hasattr(self, "base_expert"):
            return None

        # Check if ANY gradients exist
        has_any_grads = False
        total_params = 0
        for param in self.base_expert.parameters():
            total_params += 1
            if param.grad is not None:
                has_any_grads = True

        if not has_any_grads:
            if not hasattr(self, "_no_grads_warned"):
                print(
                    f"[Prismatic] log_gradient_dynamics: No gradients found ({total_params} params checked)"
                )
                self._no_grads_warned = True
            return None

        # Accumulate gradient norms by tier
        tier_grads = {"top": [], "bottom": [], "middle": []}

        for param_name, param in self.base_expert.named_parameters():
            if param.grad is None:
                continue

            # Compute tier masks based on weight magnitudes
            num_params = param.numel()
            num_per_side = max(1, int(num_params * self.sparsity / 2))
            flat_param = param.flatten().abs()
            flat_grad = param.grad.flatten().abs()

            # Top tier: highest magnitude weights
            top_threshold = torch.topk(flat_param, num_per_side).values[-1]
            is_top = flat_param >= top_threshold

            # Bottom tier: lowest magnitude non-zero weights
            non_zero_mask = flat_param > 0
            if non_zero_mask.sum() >= num_per_side:
                non_zero_param = flat_param[non_zero_mask]
                bottom_threshold = torch.topk(
                    non_zero_param, num_per_side, largest=False
                ).values[-1]
                is_bottom = (flat_param <= bottom_threshold) & non_zero_mask
            else:
                is_bottom = non_zero_mask

            # Middle tier: everything else
            is_middle = ~(is_top | is_bottom)

            # Collect gradient norms for each tier
            if is_top.sum() > 0:
                tier_grads["top"].extend(flat_grad[is_top].tolist())
            if is_bottom.sum() > 0:
                tier_grads["bottom"].extend(flat_grad[is_bottom].tolist())
            if is_middle.sum() > 0:
                tier_grads["middle"].extend(flat_grad[is_middle].tolist())

        # Compute statistics for each tier
        grad_summary = {}
        for tier, grad_list in tier_grads.items():
            if grad_list:
                # L2 norm across all gradients in this tier
                tier_norm = sum(g**2 for g in grad_list) ** 0.5
                grad_summary[f"{tier}_norm"] = tier_norm
                grad_summary[f"{tier}_max"] = max(grad_list)
                grad_summary[f"{tier}_min"] = min(grad_list)
                grad_summary[f"{tier}_mean"] = sum(grad_list) / len(grad_list)

        self._dynamics_metrics = {
            "base_expert_gradients": grad_summary,
            "num_experts": self.num_experts,
            "perturbation_mode": self.perturbation_mode,
            "perturbation_strategy": self.perturbation_strategy,
        }

        return self._dynamics_metrics

    def get_metrics(self) -> dict:
        """Return collected metrics for logging."""
        return self._metrics.copy()

    def get_dynamics_metrics(self) -> dict:
        """Return collected dynamics/gradient metrics for logging."""
        return self._dynamics_metrics.copy()

    def _is_router_mode(self, args: tuple, kwargs: dict) -> bool:
        """Check if we're in router mode based on arguments."""
        return len(args) == 7 or "layer" in kwargs

    def _parse_router_args(self, args: tuple, kwargs: dict) -> tuple:
        """Parse arguments for router mode."""
        if len(args) == 7:
            return args
        else:
            return (
                kwargs["layer"],
                kwargs["inputs"],
                kwargs.get("attention_mask"),
                kwargs.get("past_key_values"),
                kwargs.get("current_state"),
                kwargs.get("current_depth", 0),
                kwargs.get("block_ids"),
            )

    def _parse_direct_args(
        self, args: tuple, kwargs: dict
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parse arguments for direct mode."""
        if len(args) >= 2:
            return args[0], args[1]
        elif len(args) == 1:
            return args[0], None
        else:
            inputs = kwargs.get("inputs")
            if inputs is None:
                raise ValueError(f"No inputs provided. Args: {args}, Kwargs: {kwargs}")
            return inputs, kwargs.get("current_state")
