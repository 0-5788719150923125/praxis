"""Distance (praxis/routers/distance.py): SMEAR plus a parameter-distance loss
that pushes the expert deviations apart."""

import torch
import torch.nn as nn

from praxis.routers.distance import Distance
from tests.routers.toy_block import make


def test_distance_pushes_expert_deviations_apart():
    """Zero at init, where every deviation is zero; negative once they move,
    pushing each expert away from expert 0; and it reaches the deviations."""
    router, _ = make(Distance)
    router.train()
    assert router.diversity_loss().item() == 0.0
    with torch.no_grad():
        for w in router.wrappers.values():
            nn.init.normal_(w.lora_b, std=0.1)
        for p in router.deltas.values():
            nn.init.normal_(p, std=0.1)
    aux = router.router_aux_loss()
    assert set(aux) == {"distance_diversity"}
    assert aux["distance_diversity"].item() < 0.0
    aux["distance_diversity"].backward()
    assert router.wrappers["attn_qkv"].lora_b.grad.abs().sum() > 0
    router.eval()
    assert router.router_aux_loss() == {}
