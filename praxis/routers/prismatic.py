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
        helical_modulation: Apply helical/spiral modulation using Euler's formula (default: True)
            When True (DEFAULT): perturbations modulated by cos(2π·position/wavelength + phase)
            When False: simple deterministic perturbations (clean baseline for ablation)
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
    helical_modulation: bool = True  # Structure transfer (Euler's formula)
    helical_wavelength: float = 3141.592653589793  # π × 1000 (using actual π value)


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
        self.helical_modulation = getattr(config, "helical_modulation", True)
        self.helical_wavelength = getattr(
            config, "helical_wavelength", 3141.592653589793
        )

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
            # creates its own perturbed clones from the base expert

        # Create expert clones with perturbations
        self.experts = self._create_perturbed_experts(base_expert)
        self.experts = nn.ModuleList(self.experts)

        # Router network with layer normalization (following SMEAR design)
        self.router_norm = nn.LayerNorm(self.hidden_size)
        self.router = nn.Linear(self.hidden_size, len(self.experts))

        # Track parameter names for merging
        self.parameter_names: List[str] = []

        # Metrics storage for convergence tracking
        self._metrics = {}

        # Dynamics metrics storage (gradient tracking)
        self._dynamics_metrics = {}

        # Cache for weight tiers (top/bottom/middle) - computed once during init
        self._weight_tiers = self._compute_weight_tiers()

    def _create_perturbed_experts(self, base_expert: nn.Module) -> List[nn.Module]:
        """
        Create N expert clones from base expert with static perturbations.

        Expert 0: Clean (unperturbed) - the "right eye" baseline
        Experts 1+: Perturbed clones - the "left eye(s)" with forced irregularities

        The perturbations are deterministic (hash-based) and static (not trainable),
        forcing each expert to traverse different regions of the computational substrate
        throughout training.

        Args:
            base_expert: The base module to clone and perturb

        Returns:
            List of expert modules (1 clean + N-1 perturbed)
        """
        experts = []

        for expert_idx in range(self.num_experts):
            # Deep copy the base expert
            expert = copy.deepcopy(base_expert)

            if expert_idx == 0:
                # Expert 0: Clean, unperturbed - consensus reality
                experts.append(expert)
                continue

            # Experts 1+: Apply static perturbations - alternative realities
            self._apply_static_perturbations(expert, expert_idx)
            experts.append(expert)

        return experts

    def _apply_static_perturbations(self, expert: nn.Module, expert_idx: int) -> None:
        """
        Apply deterministic sparse perturbations to an expert's parameters.

        This implements the core "prismatic" mechanism: forcing the expert to adapt
        to an irregular parameter space. The perturbations are:

        1. Deterministic: pi-based (directional) or hash-based (noise)
        2. Sparse: Only perturb top-k% of weights (by magnitude or random)
        3. Adaptive: Scaled by parameter magnitude to preserve structure
        4. Static: Applied once, not trainable - architectural constraints

        Why Sparse Perturbations?
        ------------------------
        Connection to Lottery Ticket Hypothesis (Frankle & Carbin, 2019):

        LTH shows that sparse critical subnetworks exist in dense networks. Prismatic
        inverts this insight: instead of pruning to find winning tickets, we perturb
        sparse high-magnitude weights to create "obstacle courses" that force discovery
        of alternative computational paths.

        Like Willow's single-qubit perturbations, targeting 10% of weights creates just
        enough structural irregularity to force genuine exploration without destroying
        functionality. The unperturbed 90% provide stability; the perturbed 10% force
        diversity.

        Why Simple Deterministic Perturbations?
        ---------------------------------------
        Phase 1 (baseline): Use clean, simple perturbations without decoration.
        Each expert gets a fixed transformation that's reproducible.

        Phase 2 (experiment): Test if helical modulation (Euler's formula) transfers
        structure into learned patterns. This is an optional experiment to test whether
        external perturbation structure influences internal gradient/feature patterns.

        Args:
            expert: The expert module to perturb (modified in-place)
            expert_idx: Expert index for phase offset (if helical modulation enabled)
        """
        for param_name, param in expert.named_parameters():
            if not param.requires_grad:
                # Skip frozen parameters
                continue

            # Note: We DO perturb normalization parameters (LayerNorm, etc.)
            # This forces each expert to learn different normalization strategies
            # in response to the perturbed computational substrate.
            # The network must adapt its rebalancing, not rely on identical norms.

            # Create RNG generator for noise mode and random mask
            # For directional modes: generator not used (deterministic)
            # For noise mode: provides reproducible random noise
            generator = torch.Generator(device=param.device).manual_seed(expert_idx)

            # Create sparse mask: which weights to perturb
            mask = self._create_sparse_mask(param, generator)

            # Generate adaptive perturbation
            # For directional modes: simple deterministic (optionally modulated by helical pattern)
            # For noise mode: uses seeded RNG
            perturbation = self._generate_perturbation(
                param, mask, generator, expert_idx, param_name
            )

            # Apply perturbation (in-place, static)
            # This is NOT part of the computational graph - it's an architectural constraint
            with torch.no_grad():
                param.add_(perturbation)

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

        1. "attractive" mode (DEFAULT - Neuronal Regeneration):
            Phase 1 (helical_modulation=False):
                ε = -perturbation_scale * W * mask
                W_new = W + ε
                Clean perturbations - suppress top, amplify bottom
                Attracts attention to dormant neurons

            Phase 2 (helical_modulation=True):
                ε = -perturbation_scale * W * mask * spiral(expert_idx)
                W_new = W + ε
                Tests if helical structure transfers into learned patterns

        2. "repulsive" mode (Extreme Exploration):
            Phase 1: ε = perturbation_scale * W * mask
            Phase 2: ε = perturbation_scale * W * mask * spiral(expert_idx)
            Repels weights to numerical extremes

        3. "noise" mode (legacy):
            ε = perturbation_scale * |W| * N(0,1) * |mask|
            Random Gaussian noise (not affected by helical modulation)

        Helical Modulation (Phase 2):
        -----------------------------
        When enabled, uses Euler's formula to create spiral patterns:
            spiral = cos(2π * position / wavelength + phase_offset)

        Each expert gets different phase offset: phase = expert_idx * 2π / num_experts
        This creates harmonic relationships between experts.

        **Hypothesis**: External helical structure in perturbations may transfer into
        internal patterns (gradients, learned features). This is testable by comparing
        Phase 1 vs Phase 2 results.

        Args:
            param: Original parameter tensor
            mask: Tiered mask (+1 top, -1 bottom, 0 middle) or binary mask
            generator: Seeded random generator (used only for noise mode)
            expert_idx: Expert index for phase offset
            param_name: Parameter name (unused, kept for interface compatibility)

        Returns:
            Perturbation tensor (same shape as param)
        """
        import math

        if (
            self.perturbation_mode == "repulsive"
            or self.perturbation_mode == "attractive"
        ):
            # Phase 1: Simple deterministic perturbations (clean baseline)
            if self.perturbation_mode == "attractive":
                # Suppress top, amplify bottom (attract to dormant)
                base_perturbation = -self.perturbation_scale * param * mask
            else:
                # Amplify top, suppress bottom (repel to extremes)
                base_perturbation = self.perturbation_scale * param * mask

            # Phase 2: Optionally modulate with helical pattern
            if self.helical_modulation and expert_idx > 0:
                # Create spiral modulation using Euler's formula
                # spiral = cos(2π * position / wavelength + phase_offset)

                # Spatial positions for each weight
                num_weights = param.numel()
                positions = torch.arange(
                    num_weights, dtype=param.dtype, device=param.device
                )

                # Normalize to wavelength
                positions_normalized = 2 * math.pi * positions / self.helical_wavelength

                # Expert-specific phase offset (harmonic relationships)
                phase_offset = expert_idx * 2 * math.pi / self.num_experts

                # Create spiral pattern
                spiral = torch.cos(positions_normalized + phase_offset)
                spiral = spiral.reshape(param.shape)

                # Modulate perturbation with spiral
                perturbation = base_perturbation * spiral
            else:
                # No helical modulation - clean baseline
                perturbation = base_perturbation
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
        modulation = "helical" if self.helical_modulation else "clean"
        return (
            f"{self.__class__.__name__}("
            f"num_experts={len(self.experts)}, "
            f"strategy={self.perturbation_strategy}, "
            f"mode={self.perturbation_mode}, "
            f"scale={self.perturbation_scale}, "
            f"sparsity={self.sparsity}, "
            f"{modulation})"
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

        # Use first expert as base structure
        base_module = self.experts[0]

        # Apply merged parameters
        forward_args = (
            inputs,
            attention_mask,
            past_key_values,
            current_state,
            current_depth,
            block_ids,
        )

        result = torch.func.functional_call(
            base_module, merged_state_dict, forward_args, {}
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
        base_module = self.experts[0]

        result = torch.func.functional_call(
            base_module, merged_state_dict, (inputs, current_state), {}
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

        This implements SMEAR's core mechanism: instead of routing inputs to
        different experts, we merge the expert parameters themselves based on
        routing probabilities.

        Mathematical formulation:
            W_merged = Σ_i (p_i * W_expert_i)

        where p_i are routing probabilities and W_expert_i are perturbed parameters.

        Connection to Prismatic Attention:
        ---------------------------------
        This merging adaptively combines different "eyes" (perturbed experts).
        Each eye has traversed a different computational substrate due to static
        perturbations. The routing learns which combination of perspectives is
        most useful for the current input.

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

        # Compute weight divergence (cosine similarity) every forward pass
        self._log_weight_divergence()

        # Collect parameter names from base expert
        self.parameter_names = self._collect_parameter_names(self.experts[0])

        for param_name in self.parameter_names:
            merged_param: Optional[torch.Tensor] = None

            for expert_idx, expert in enumerate(self.experts):
                param = self._get_module_parameter(expert, param_name)

                if param is None:
                    raise ValueError(
                        f"Parameter '{param_name}' not found in expert {expert_idx}"
                    )

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

    def _log_weight_divergence(self) -> None:
        """
        Compute cosine similarity between base expert and perturbed experts.

        Measures directional divergence in weight space - how much each expert's
        parameters point in different directions from the clean baseline.

        Cosine similarity = 1.0 means identical direction (no divergence)
        Cosine similarity = 0.0 means orthogonal (maximal divergence)
        Cosine similarity = -1.0 means opposite direction

        This is computed every forward pass to track runtime weight expression,
        independent of gradient dynamics or training updates.
        """
        if len(self.experts) < 2:
            return

        try:
            # Get base expert parameters (flattened)
            base_params = []
            for param in self.experts[0].parameters():
                if param.requires_grad:
                    base_params.append(param.flatten())

            if not base_params:
                return

            base_vector = torch.cat(base_params)

            # Compute cosine similarity for each perturbed expert
            for expert_idx in range(1, len(self.experts)):
                expert_params = []
                for param in self.experts[expert_idx].parameters():
                    if param.requires_grad:
                        expert_params.append(param.flatten())

                expert_vector = torch.cat(expert_params)

                # Cosine similarity: (A · B) / (||A|| × ||B||)
                cosine_sim = torch.nn.functional.cosine_similarity(
                    base_vector.unsqueeze(0), expert_vector.unsqueeze(0), dim=1
                ).item()

                # Store as metric (will be logged via extra_metrics)
                self._metrics[f"expert_{expert_idx}_cosine_similarity"] = cosine_sim

                # Also store angular divergence in degrees (more interpretable)
                # angle = arccos(similarity)
                angle_rad = torch.acos(torch.tensor(cosine_sim).clamp(-1.0, 1.0))
                angle_deg = angle_rad * 180.0 / 3.141592653589793
                self._metrics[f"expert_{expert_idx}_weight_angle"] = angle_deg.item()

        except Exception as e:
            # Silently fail - don't break forward pass
            pass

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

    def _compute_weight_tiers(self) -> Dict[str, Dict[str, str]]:
        """
        Compute which weights belong to top/bottom/middle tiers for each expert.

        This is done once at initialization based on the perturbation masks used during
        expert creation. Returns a mapping of expert_idx -> param_name -> tier.

        Tiers:
        - 'top': Top sparsity/2 weights by magnitude (if dual_sided)
        - 'bottom': Bottom sparsity/2 weights by magnitude (if dual_sided)
        - 'middle': Everything else (unperturbed)

        Returns:
            Dict mapping expert indices to parameter tiers
        """
        weight_tiers = {}

        for expert_idx, expert in enumerate(self.experts):
            tiers = {}

            for param_name, param in expert.named_parameters():
                if not param.requires_grad:
                    continue

                # Generate same generator used during perturbation (for random strategy)
                generator = torch.Generator(device=param.device).manual_seed(expert_idx)

                # Determine tier based on weight magnitudes
                # This applies to ALL experts so we can compare gradient dynamics
                num_params = param.numel()
                num_per_side = max(1, int(num_params * self.sparsity / 2))
                flat_param = param.flatten().abs()

                # Top tier: highest magnitude weights
                top_threshold = torch.topk(flat_param, num_per_side).values[-1]
                is_top = (flat_param >= top_threshold).reshape(param.shape)

                # Bottom tier: lowest magnitude non-zero weights
                non_zero_param = flat_param[flat_param > 0]
                if len(non_zero_param) >= num_per_side:
                    bottom_threshold = torch.topk(
                        non_zero_param, num_per_side, largest=False
                    ).values[-1]
                    is_bottom = (
                        (flat_param <= bottom_threshold) & (flat_param > 0)
                    ).reshape(param.shape)
                else:
                    is_bottom = (flat_param > 0).reshape(param.shape)

                # For Expert 0: Categorize all params into top/bottom/middle tiers
                # For Expert 1+: Categorize based on where perturbations landed
                if expert_idx == 0:
                    # Categorize the entire parameter by its predominant tier
                    top_count = is_top.sum().item()
                    bottom_count = is_bottom.sum().item()

                    if top_count > 0:
                        tiers[param_name] = "top"
                    elif bottom_count > 0:
                        tiers[param_name] = "bottom"
                    else:
                        tiers[param_name] = "middle"
                else:
                    # For perturbed experts, check if perturbation mask overlaps with tiers
                    mask = self._create_sparse_mask(param, generator)

                    if self.perturbation_strategy == "dual_sided":
                        # Determine which tier was perturbed
                        if (mask * is_top).sum() > 0:
                            tiers[param_name] = "top"
                        elif (mask * is_bottom).sum() > 0:
                            tiers[param_name] = "bottom"
                        else:
                            tiers[param_name] = "middle"
                    else:
                        # For non-dual-sided, mark as perturbed or middle
                        if mask.sum() > 0:
                            tiers[param_name] = "perturbed"
                        else:
                            tiers[param_name] = "middle"

            weight_tiers[expert_idx] = tiers

        return weight_tiers

    def log_gradient_dynamics(self) -> Optional[Dict]:
        """
        Log expert gradient dynamics by weight tier.

        This captures the core question: Do clean vs perturbed experts learn differently?
        Specifically for dual-sided perturbation: Are bottom weights actually waking up?

        Should be called after backward() but before optimizer.step().

        Returns:
            Dict with expert gradients by tier, or None if no gradients available
        """
        if not hasattr(self, "experts") or len(self.experts) == 0:
            print(f"[Prismatic] log_gradient_dynamics: No experts found")
            return None

        # Check if ANY gradients exist
        has_any_grads = False
        grad_count = 0
        total_params = 0
        for expert in self.experts:
            for param in expert.parameters():
                total_params += 1
                if param.grad is not None:
                    has_any_grads = True
                    grad_count += 1

        if not has_any_grads:
            # No gradients available - backward() wasn't called or gradients were zeroed
            if not hasattr(self, "_no_grads_warned"):
                print(
                    f"[Prismatic] log_gradient_dynamics: No gradients found ({total_params} params checked)"
                )
                self._no_grads_warned = True
            return None

        expert_grads = {}

        for expert_idx, expert in enumerate(self.experts):
            # Accumulate gradient norms by tier (element-level, not parameter-level)
            tier_grads = {"top": [], "bottom": [], "middle": []}

            for param_name, param in expert.named_parameters():
                if param.grad is None:
                    continue

                # Compute tier masks for this parameter based on weight magnitudes
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
            expert_grad_summary = {}
            for tier, grad_list in tier_grads.items():
                if grad_list:
                    # L2 norm across all gradients in this tier
                    tier_norm = sum(g**2 for g in grad_list) ** 0.5
                    expert_grad_summary[f"{tier}_norm"] = tier_norm
                    expert_grad_summary[f"{tier}_max"] = max(grad_list)
                    expert_grad_summary[f"{tier}_min"] = min(grad_list)

            expert_grads[f"expert_{expert_idx}"] = expert_grad_summary

        # Debug: show what we collected on first call only
        if not hasattr(self, "_grad_log_count"):
            self._grad_log_count = 0
        self._grad_log_count += 1

        if self._grad_log_count == 1:
            print(
                f"[Prismatic] Gradient dynamics collection started for {len(self.experts)} experts"
            )
            for exp_key, metrics in expert_grads.items():
                print(f"  {exp_key}: {list(metrics.keys())}")

        # Calculate gradient-based divergence between Expert 0 and others
        gradient_divergence = {}
        if "expert_0" in expert_grads:
            clean_norms = expert_grads["expert_0"]
            for expert_idx in range(1, len(self.experts)):
                perturbed_norms = expert_grads.get(f"expert_{expert_idx}", {})

                # Simple divergence: mean absolute difference across tiers
                diffs = []
                for key in ["top_norm", "bottom_norm", "middle_norm"]:
                    if key in clean_norms and key in perturbed_norms:
                        diffs.append(abs(clean_norms[key] - perturbed_norms[key]))

                if diffs:
                    gradient_divergence[f"expert_{expert_idx}_divergence"] = sum(
                        diffs
                    ) / len(diffs)

        # Calculate weight-level divergence (architectural corruption from perturbations)
        # This measures L2 distance between actual parameters of clean vs perturbed experts
        weight_divergence = {}
        for expert_idx in range(1, len(self.experts)):
            total_diff = 0.0
            num_params = 0

            # Compare all parameters between expert 0 (clean) and expert i (perturbed)
            for (name0, param0), (name_i, param_i) in zip(
                self.experts[0].named_parameters(),
                self.experts[expert_idx].named_parameters(),
            ):
                if not param0.requires_grad:
                    continue

                # L2 norm of weight difference (measures corruption)
                param_diff = (param0 - param_i).pow(2).sum().sqrt().item()
                total_diff += param_diff
                num_params += 1

            if num_params > 0:
                weight_divergence[f"expert_{expert_idx}_weight_divergence"] = (
                    total_diff / num_params
                )

        self._dynamics_metrics = {
            "expert_gradients": expert_grads,
            "divergence_scores": gradient_divergence,
            "weight_divergence": weight_divergence,
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
