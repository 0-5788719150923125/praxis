import itertools
from enum import Enum
from typing import List

import pytest
import torch
from torch import nn

from praxis import PraxisConfig
from praxis.residuals import RESIDUAL_REGISTRY

MODULE_CLASSES = list(RESIDUAL_REGISTRY.values())


# Define test parameters in a more structured way
TEST_PARAMS = {
    "hidden_sizes": [64, 128, 256],
    "num_heads": [1, 2, 3],
}


def get_residual_configs() -> List[PraxisConfig]:
    """Generate valid attention configurations using itertools.product."""
    return [
        PraxisConfig(hidden_size=hidden_size, num_heads=num_heads)
        for hidden_size, num_heads in itertools.product(
            TEST_PARAMS["hidden_sizes"], TEST_PARAMS["num_heads"]
        )
    ]


@pytest.fixture(params=list(itertools.product(MODULE_CLASSES, get_residual_configs())))
def module_setup(request, config):
    """
    Parametrized fixture that provides module and its configuration.

    Args:
        request: pytest request object containing the parameter tuple
        config: the base config fixture from conftest.py

    Returns:
        tuple: (module instance, config)
    """
    module_class, res_config = request.param

    setattr(config, "hidden_size", res_config.hidden_size)
    setattr(config, "num_heads", res_config.num_heads)

    module = module_class(config.hidden_size)
    return module, res_config


def test_forward_pass(module_setup):
    """Test forward pass with valid parameter combinations."""
    module, res_config = module_setup
    batch_size = 32
    seq_len = 16

    # Create input tensor
    x = torch.randn(batch_size, seq_len, res_config.hidden_size)

    # Run forward pass
    residual, beta = module.connect_width(x)
    y = nn.Identity()(module.format_state(residual))
    merged = module.connect_depth(residual, y, beta)
    final = module.format_state(merged)

    # Verify output shape
    assert final.shape == (batch_size, seq_len, res_config.hidden_size)


# --- SMEAR residual router (praxis/residuals/smear.py) ----------------------


def test_smear_recovers_pure_styles_at_the_extremes():
    """Pushing the router fully to one style reproduces that style exactly:
    all-standard = x + branch; all-rezero with zero-init alpha = identity."""
    from praxis.residuals import SmearResidual

    torch.manual_seed(0)
    mixer = SmearResidual(8, num_depths=4)
    x = torch.randn(2, 5, 8)
    branch = torch.randn(2, 5, 8)

    with torch.no_grad():
        mixer.logits.fill_(0.0)
        mixer.logits[:, 0] = 20.0  # all standard
    assert torch.allclose(mixer.connect_depth(x, branch, None, 1), x + branch)

    with torch.no_grad():
        mixer.logits.fill_(0.0)
        mixer.logits[:, 1] = 20.0  # all rezero (alpha zero-init -> identity)
    assert torch.allclose(mixer.connect_depth(x, branch, None, 1), x)


def test_smear_uniform_init_half_gain_and_depth_indexing():
    """Zero-init logits = uniform mix: with zero-init rezero alpha the entry
    gain is 0.5. Distinct per-depth logits produce distinct outputs."""
    from praxis.residuals import SmearResidual

    torch.manual_seed(0)
    mixer = SmearResidual(8, num_depths=4)
    x = torch.randn(2, 5, 8)
    branch = torch.randn(2, 5, 8)
    assert torch.allclose(
        mixer.connect_depth(x, branch, None, 0), x + 0.5 * branch, atol=1e-6
    )

    with torch.no_grad():
        mixer.logits[2, 0] = 5.0  # depth 2 leans standard
    d0 = mixer.connect_depth(x, branch, None, 0)
    d2 = mixer.connect_depth(x, branch, None, 2)
    assert not torch.allclose(d0, d2)


def test_smear_router_and_children_receive_gradient():
    from praxis.residuals import SmearResidual
    from praxis.residuals.rezero import ReZeroConnection

    torch.manual_seed(0)
    mixer = SmearResidual(8, num_depths=4)
    x = torch.randn(2, 5, 8, requires_grad=True)
    branch = torch.randn(2, 5, 8)
    mixer.connect_depth(x, branch, None, 1).sum().backward()
    assert mixer.logits.grad is not None and mixer.logits.grad.abs().sum() > 0
    rezero = [m for m in mixer.mix if isinstance(m, ReZeroConnection)][0]
    assert rezero.alpha.grad is not None


def test_smear_rejects_hyper():
    """Hyper widens the residual state to rate streams - a different state
    contract - so the mixer refuses it rather than silently mis-blending."""
    from praxis.residuals import SmearResidual

    with pytest.raises(ValueError, match="stream"):
        SmearResidual(8, num_depths=4, styles=("standard", "hyper"))


def test_smear_block_integration_and_metrics():
    """A transformer block built with residual_type=smear trains its router,
    and the decoder surfaces per-depth style shares."""
    from praxis.blocks.transformer import TransformerBlock
    from praxis.modeling import PraxisForCausalLM
    from praxis.residuals import SmearResidual

    torch.manual_seed(0)
    cfg = PraxisConfig(
        vocab_size=1000,
        hidden_size=32,
        embed_size=32,
        num_heads=4,
        num_layers=2,
        depth=4,
        decoder_type="sequential",
        residual_type="smear",
    )
    block = TransformerBlock(cfg)
    assert isinstance(block.attn_res, SmearResidual)
    x = torch.randn(2, 16, 32)
    out, _, _, _ = block(x, attention_mask=None, current_depth=1)
    out.sum().backward()
    assert block.attn_res.logits.grad is not None
    assert block.ffn_res.logits.grad is not None

    shares = block.attn_res.style_shares()
    assert shares["residual/mix_rezero_d0"] == pytest.approx(0.5)
    assert len(shares) == 4  # one rezero share per depth

    model = PraxisForCausalLM(cfg)
    assert "residual/mix_rezero_d0" in model.decoder.get_metrics()
