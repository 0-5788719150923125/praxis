"""VEAR (praxis/routers/vear.py): SMEAR plus sharpened routing and
inter-expert repulsion - Praxis's own departure from the paper."""

import torch

from praxis.routers.smear import SMEAR
from praxis.routers.vear import VEAR
from tests.routers.toy_block import Cfg, make


def test_sharpening_concentrates_without_changing_shape():
    """The only check that VEAR's SHARPEN reaches the merge itself; the selection
    metrics re-apply it on their own and cannot show this."""
    sharp, _ = make(VEAR)
    plain, _ = make(SMEAR)
    sharp.router.load_state_dict(plain.router.state_dict())
    sharp.router_norm.load_state_dict(plain.router_norm.state_dict())
    with torch.no_grad():  # a non-uniform router, so sharpening has something to do
        torch.nn.init.normal_(plain.router.weight, std=0.5)
        sharp.router.weight.copy_(plain.router.weight)
    x = torch.randn(8, 6, Cfg.hidden_size)
    ws, _ = sharp._coefficients(x, 0)
    wp, _ = plain._coefficients(x, 0)
    assert ws.shape == wp.shape
    assert ws.max(dim=-1).values.mean() > wp.max(dim=-1).values.mean()


def test_repulsion_is_a_scalar_and_training_only():
    """Repulsion is VEAR's, not SMEAR's - the paper's SMEAR has neither it nor
    sharpening."""
    assert not hasattr(SMEAR, "router_aux_loss")
    router, _ = make(VEAR)
    router.train()
    aux = router.router_aux_loss()
    assert "vear_repulsion" in aux and aux["vear_repulsion"].dim() == 0
    router.eval()
    assert router.router_aux_loss() == {}
