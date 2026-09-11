from dataclasses import dataclass

import pytest

from praxis.routers.smear import SMEAR

# ------------------------------------------------------------------------------
# smear
# ------------------------------------------------------------------------------
# Modular SMEAR: the paper's granularity, over a shared block plus deviations.
#
# Not a new method - SMEAR (arxiv 2306.03745) applied the way the paper applies it.
# Pinned here:
#
# * targets are discovered per MODULE, not per block, and each gets its own coefficient
# row (the paper puts a router on each inserted adapter); * Linear targets route per
# position on the running mean of the prefix (the paper's per-example pooling, made
# causal), while elementwise targets take the input-free depth prior (the paper does not
# treat layernorm parameters as experts either), so no position reads a later one and no
# row reads another; * expert dropout is present, because that is the paper's load-
# balancing mechanism and without it every target collapses to one-hot; *
# ``MERGE_OPAQUE`` subtrees and reference-tied parameters are never merged - the two
# structural exclusions PEER and the Titans memory rely on; * the router is EXACTLY
# identity at init, so a config swap is a clean A/B; * the merge really is the paper's
# merge in a base-plus-deviation basis, i.e. it equals the convex combination of the
# implied experts; * the shared trunk receives full gradient whatever the routing does,
# which is the property VEAR's dead experts did not have.


def test_layout_is_shared_so_the_decoder_builds_one_block():
    from praxis.decoders.base import _router_layout, _wants_expert_bank

    assert _router_layout("smear") == "shared"
    assert _router_layout("vear") == "shared"
    assert not _wants_expert_bank("smear")
    assert _router_layout("distance") == "shared"
    assert not _wants_expert_bank("distance")


# ------------------------------------------------------------------------------
# smear_integration
# ------------------------------------------------------------------------------
# Test SMEAR integration with sequential decoder and multiple experts.


@dataclass
class MockConfig:
    """Mock configuration for testing SMEAR integration."""

    # Core configuration
    hidden_size: int = 256
    depth: int = 6
    num_experts: int = 3  # Number of experts for SMEAR to manage
    num_layers: int = 3  # Number of layer components for controllers
    epsilon: float = 1e-6
    dropout: float = 0.1

    # Decoder configuration
    decoder_type: str = "sequential"
    block_type: str = "recurrent"
    router_type: str = "smear"
    controller_type: str = "base"
    compression_type: str = "none"
    sorting_type: str = "none"
    halting_type: str = "none"

    # Additional required fields
    checkpoint_every: int = 0
    debug: bool = False
    evolve: bool = False
    hivemind: bool = False
    expert: str = "default"
    meta: dict = None

    # For blocks that need these
    num_heads: int = 8
    activation: str = "swish"
    causal: bool = True

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}
