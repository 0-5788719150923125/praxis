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

Pi-Digit Seeding (Quantum Echoes Through Mathematical Constants):
-----------------------------------------------------------------
Instead of arbitrary random seeds, perturbations are seeded by walking backwards through
the digits of pi (3.14159265358979...). Each expert gets a single digit (0-9) from pi's
infinite sequence:

- Expert 0: Clean (unperturbed) - the original quantum state at t=0
- Expert 1: Seeded by pi_digit[position - 1] - archive/lag/echo from t-1
- Expert 2: Seeded by pi_digit[position - 2] - archive/lag/echo from t-2
- Expert 3: Seeded by pi_digit[position - 3] - archive/lag/echo from t-3
- ...

Walking BACKWARDS creates a temporal lag structure - each expert is an archived copy
"behind" the current state, corrupted by where pi was at increasingly distant positions.
During iterative reasoning (depth>layers), the model learns to integrate across these
temporal lags, building a fuzzy state machine where it can blend {current, t-1, t-2, t-3}.

Like djent polyrhythms (TOOL, Meshuggah): deterministically unpredictable, yet perfectly
reproducible. Pi's digits create phase-shifted perturbations - the same complex rhythm
experienced at different temporal offsets.

The backwards walk enables the model to learn a temporal gradient during training,
then exploit that structure during inference. Efficient caching (computed once) makes
this practically costless.

Key Insight: "Train on consensus, you manifest the lowest common denominator."
Static perturbations prevent convergence to consensus, maintaining genuine diversity throughout
training. Each expert traverses a fundamentally different computational substrate.
"""

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mpmath import mp

# Module-level cache for pi digits
# Compute once, slice forever - supports efficient backwards walk
_PI_CACHE: Optional[str] = None
_PI_CACHE_START: int = 0
_PI_CACHE_END: int = 0


def _ensure_pi_cache(start_position: int, end_position: int) -> None:
    """
    Ensure pi cache covers the range [start_position, end_position].

    Computes pi digits once and caches them for efficient repeated access.
    This enables efficient backwards walk through pi without recomputation.

    Args:
        start_position: First position needed (inclusive)
        end_position: Last position needed (inclusive)
    """
    global _PI_CACHE, _PI_CACHE_START, _PI_CACHE_END

    # Check if we need to extend the cache
    if (
        _PI_CACHE is None
        or start_position < _PI_CACHE_START
        or end_position > _PI_CACHE_END
    ):
        # Compute with buffer to reduce future recomputations
        buffer = 10000
        new_start = (
            min(start_position, _PI_CACHE_START if _PI_CACHE else start_position)
            - buffer
        )
        new_end = (
            max(end_position, _PI_CACHE_END if _PI_CACHE else end_position) + buffer
        )
        new_start = max(0, new_start)  # Don't go below 0

        # Compute pi digits for the entire range
        num_digits_needed = new_end + 1
        mp.dps = num_digits_needed + 100  # Extra precision for safety
        pi_str = str(mp.pi).replace(".", "")

        # Cache the computed digits
        _PI_CACHE = pi_str[:num_digits_needed]
        _PI_CACHE_START = 0
        _PI_CACHE_END = len(_PI_CACHE) - 1


def get_pi_digit_at(position: int) -> int:
    """
    Get a single digit from pi at the specified position (cached, efficient).

    Position 0 = 3, position 1 = 1, position 2 = 4, etc.

    Uses a module-level cache to avoid recomputing pi digits.
    Particularly efficient for backwards walks where we access
    sequential positions like [99999, 99998, 99997, ...].

    Args:
        position: Index into pi's digit sequence

    Returns:
        Single digit (0-9) at that position
    """
    # Ensure cache covers this position
    _ensure_pi_cache(position, position)

    # Return cached digit
    return int(_PI_CACHE[position])


@dataclass
class PrismaticConfig:
    """Configuration for Prismatic attention module.

    Attributes:
        hidden_size: Hidden dimension size
        num_experts: Number of expert clones (including 1 clean expert)
        perturbation_scale: Scale factor for perturbations (default: 1.0)
            Adds Gaussian noise with std = scale * |W|
            1.0 = add noise equal to weight magnitude (aggressive corruption)
            0.1 = add 10% noise (moderate)
            0.01 = add 1% noise (conservative)
        sparsity: Fraction of weights to perturb (default: 0.1 = 10%)
        perturb_by_magnitude: If True, perturb by magnitude; else random
        perturbation_strategy: Strategy for selecting weights to perturb
            - "dual_sided": Top sparsity/2 + bottom sparsity/2 by magnitude (default)
            - "top_only": Top sparsity by magnitude (original approach)
            - "bottom_only": Bottom sparsity by magnitude
            - "random": Random sparsity selection
        dropout: Dropout probability for expert dropout during training
        use_pi_seeding: If True, use pi-digits for seeding (Quantum Echoes); else use hash
        pi_position: Starting position in pi's digit sequence (default: 100000)
    """

    hidden_size: int
    num_experts: int
    perturbation_scale: float = 1.0
    sparsity: float = 0.1
    perturb_by_magnitude: bool = True
    perturbation_strategy: str = "dual_sided"
    dropout: float = 0.1
    use_pi_seeding: bool = True
    pi_position: int = 100000


class Prismatic(nn.Module):
    """
    Prismatic Attention: Creates architectural diversity through static perturbations.

    This module implements the "dual-stream" or "multi-eye" architecture described in
    "The Blind Watchmaker" paper. It creates N expert clones from a single base expert,
    where:

    - Expert 0 (the "right eye"): Clean, unperturbed - represents consensus reality
    - Experts 1..N (the "left eye(s)"): Statically perturbed - forced alternative realities

    The perturbations are:
    - Deterministic (hash-based seeding for reproducibility)
    - Sparse (only k% of weights, focusing on high-magnitude parameters)
    - Static (not trainable - architectural constraints, not optimization targets)
    - Magnitude-aware (scaled by existing weight magnitudes)

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
        self.perturbation_scale = getattr(config, "perturbation_scale", 1.0)
        self.sparsity = getattr(config, "sparsity", 0.1)
        self.perturb_by_magnitude = getattr(config, "perturb_by_magnitude", True)
        self.perturbation_strategy = getattr(
            config, "perturbation_strategy", "dual_sided"
        )
        self.dropout_rate = getattr(config, "dropout", 0.1)
        self.use_pi_seeding = getattr(config, "use_pi_seeding", True)
        self.pi_position = getattr(config, "pi_position", 100000)

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

        1. Deterministic: hash(expert_idx || param_name) -> reproducible seed
        2. Sparse: Only perturb top-k% of weights (by magnitude or random)
        3. Adaptive: Noise scaled by parameter magnitude to preserve structure
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

        Why Magnitude-Aware Scaling?
        ---------------------------
        Preserves relative importance of parameters. Large weights receive larger
        perturbations, maintaining the network's learned prior while introducing
        controlled irregularity.

        Args:
            expert: The expert module to perturb (modified in-place)
            expert_idx: Expert index for deterministic seeding
        """
        for param_name, param in expert.named_parameters():
            if not param.requires_grad:
                # Skip frozen parameters
                continue

            # Note: We DO perturb normalization parameters (LayerNorm, etc.)
            # This forces each expert to learn different normalization strategies
            # in response to the perturbed computational substrate.
            # The network must adapt its rebalancing, not rely on identical norms.

            # Generate deterministic seed from expert_idx and param_name
            seed = self._generate_seed(expert_idx, param_name)
            generator = torch.Generator(device=param.device).manual_seed(seed)

            # Create sparse mask: which weights to perturb
            mask = self._create_sparse_mask(param, generator)

            # Generate adaptive perturbation: scaled by parameter magnitude
            # This preserves relative importance while introducing irregularity
            perturbation = self._generate_perturbation(param, mask, generator)

            # Apply perturbation (in-place, static)
            # This is NOT part of the computational graph - it's an architectural constraint
            with torch.no_grad():
                param.add_(perturbation)

    def _generate_seed(self, expert_idx: int, param_name: str) -> int:
        """
        Generate deterministic seed from expert index and parameter name.

        Two modes:

        1. Pi-Digit Seeding (Quantum Echoes) - default:
           Each expert gets seeded by walking BACKWARDS through pi's digits.
           - Expert 1: pi_digit[position - 1]
           - Expert 2: pi_digit[position - 2]
           - Expert 3: pi_digit[position - 3]
           Creates temporal lag structure where each expert is an "echo" corrupted
           by a different mathematical artifact from pi's sequence.

        2. Hash-Based Seeding (fallback):
           Uses SHA-256 for uncorrelated random seeds.

        Args:
            expert_idx: Expert index (0 is clean, 1+ are perturbed)
            param_name: Full parameter name (e.g., "layer.0.weight")

        Returns:
            Integer seed for random number generation
        """
        if self.use_pi_seeding and expert_idx > 0:
            # Walk backwards through pi's digits
            # Expert 1 gets pi[position - 1], Expert 2 gets pi[position - 2], etc.
            pi_index = self.pi_position - expert_idx
            pi_digit = get_pi_digit_at(pi_index)

            # Combine pi digit with param_name hash for diversity across parameters
            # But the pi digit provides the primary seed structure
            param_hash = hashlib.sha256(param_name.encode("utf-8")).digest()
            param_contribution = int.from_bytes(param_hash[:4], byteorder="big")

            # Seed = pi_digit (primary) + param_hash (secondary variation)
            # This gives us 10 fundamental seeds (0-9 from pi) with per-parameter variation
            seed = (pi_digit * 10**9 + param_contribution) % (2**63 - 1)

            return seed
        else:
            # Fallback to hash-based seeding
            hash_input = f"{expert_idx}||{param_name}".encode("utf-8")
            hash_digest = hashlib.sha256(hash_input).digest()
            seed = int.from_bytes(hash_digest[:8], byteorder="big")
            return seed % (2**63 - 1)

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
            Binary mask (1 = perturb, 0 = keep original)
        """
        num_params = param.numel()

        if not self.perturb_by_magnitude or self.perturbation_strategy == "random":
            # Random selection strategy
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
            top_threshold = torch.topk(flat_param, num_per_side).values[-1]
            top_mask = (flat_param >= top_threshold).float()

            # Bottom side: Lowest magnitude NON-ZERO weights
            # These live in the fine-grained precision regime
            # Near activation thresholds - small perturbations can flip dead->alive
            # May contain suppressed but useful patterns
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

            # Combine both sides (clamp to handle any overlap)
            mask = (top_mask + bottom_mask).clamp(0, 1).reshape(param.shape)

        elif self.perturbation_strategy == "top_only":
            # Original approach: Top-k by magnitude only
            num_perturbed = max(1, int(num_params * self.sparsity))
            threshold = torch.topk(flat_param, num_perturbed).values[-1]
            mask = (flat_param >= threshold).float().reshape(param.shape)

        elif self.perturbation_strategy == "bottom_only":
            # Bottom-k by magnitude (control experiment)
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
        self, param: torch.Tensor, mask: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """
        Generate adaptive perturbation scaled by parameter magnitude.

        Perturbation formula:
            ε = perturbation_scale * |W_base| * N(0,1) * mask
            W_new = W_base + ε

        Default scale = 1.0: Adds Gaussian noise with std = weight magnitude
        This creates aggressive corruption that forces exploration outside the
        learned world model. For dual-sided perturbation:
        - Top weights (large): ±100% perturbation pushes genuinely outside consensus
        - Bottom weights (small): ±100% perturbation can flip activation thresholds

        Why magnitude-aware?
        -------------------
        - Scales perturbation to weight importance
        - Bottom weights get absolute changes equal to their magnitude
        - Top weights get proportional changes that cascade through softmax
        - Both extremes get meaningful perturbations in their respective regimes

        Connection to World Model Exploration:
        -------------------------------------
        Static perturbations are architectural obstacles the model must adapt to.
        Aggressive scale (1.0) forces fundamentally different computational paths,
        not just variations within consensus. 90% unperturbed + LayerNorm provide
        stability while the 10% perturbed create genuine architectural diversity.

        Args:
            param: Original parameter tensor
            mask: Sparse binary mask indicating which weights to perturb
            generator: Seeded random generator for reproducibility

        Returns:
            Perturbation tensor (same shape as param)
        """
        # Generate standard normal noise
        noise = torch.randn(
            param.shape, dtype=param.dtype, device=param.device, generator=generator
        )

        # Scale by parameter magnitude (adaptive)
        magnitude_scale = param.abs()

        # Apply: scale * magnitude * noise * mask
        perturbation = self.perturbation_scale * magnitude_scale * noise * mask

        return perturbation

    def __repr__(self) -> str:
        seed_type = "pi-seeded" if self.use_pi_seeding else "hash-seeded"
        return (
            f"{self.__class__.__name__}("
            f"num_experts={len(self.experts)}, "
            f"strategy={self.perturbation_strategy}, "
            f"scale={self.perturbation_scale}, "
            f"sparsity={self.sparsity}, "
            f"{seed_type})"
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

    def get_metrics(self) -> dict:
        """Return collected metrics for logging."""
        return self._metrics.copy()

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
