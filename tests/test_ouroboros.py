import torch

from praxis.activations.ouroboros import MAX_STEPS, Ouroboros, drain_step_counts
from praxis.activations.serpent import Serpent
from praxis.activations.servant import Servant
from praxis.losses.regularizers import REGULARIZER_REGISTRY


def _built(cls, x, **kwargs):
    """Lazy modules materialize on first forward."""
    module = cls(**kwargs)
    module(x)
    return module


def test_eval_identity_to_serpent_at_init():
    """The -l experiment is only a controlled comparison against -k if
    Ouroboros starts as Serpent. Both gates saturate at init, so the residue is
    float rounding from writing `x + 1*(y - x)` instead of `y`."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    ouroboros = _built(Ouroboros, x, a=1.0, b=1.0, g=0.1).eval()
    serpent = _built(Serpent, x, a=1.0, b=1.0, g=0.1).eval()

    with torch.no_grad():
        assert (ouroboros(x) - serpent(x)).abs().max().item() < 1e-6


def test_budget_starts_at_one_step_and_flows_gradients():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()

    y = activation(x)
    loss = regularizer(y, torch.zeros(2, 5, dtype=torch.long))
    metrics = regularizer.training_metrics()

    # Saturated gates: one open step, the rest closed.
    assert 1.0 <= metrics["ouroboros_steps"] < 1.1
    assert metrics["ouroboros_extra_frac"] < 0.1

    (y.sum() + loss).backward()
    assert all(p.grad is not None for p in activation.parameters())

    # Init spends ~1 step against a target of 2, so the loop is UNDER budget and
    # the dual must descend (the optimizer subtracts this gradient, so it has to
    # be positive) to drive lambda negative and push depth up.
    assert regularizer.lambda_raw.grad.item() > 0


def test_dual_pushes_toward_the_target_from_both_sides():
    """The multiplier has to be SIGNED.

    A non-negative one (softplus) encodes the inequality "steps <= target",
    which is already satisfied at init, so lambda would decay to zero and a
    target above the init value would never open a single gate - the whole
    experiment would be inert. This checks the multiplier drives the constraint
    from whichever side it starts on, and that a negative lambda really does
    push the gates open.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    def dual_gradient(target, lambda_raw=0.0):
        regularizer = REGULARIZER_REGISTRY["ouroboros_budget"](target=target)
        with torch.no_grad():
            regularizer.lambda_raw.fill_(lambda_raw)
        activation = _built(Ouroboros, x).train()
        drain_step_counts()
        y = activation(x)
        regularizer(y, torch.zeros(2, 5, dtype=torch.long)).backward()
        return regularizer.lambda_raw.grad.item(), activation

    # Under budget (init ~1.03 steps, target 2.0): dual descends -> lambda < 0.
    under, activation = dual_gradient(2.0)
    assert under > 0

    # At init lambda is exactly 0 (tanh(0)), so the budget exerts NO force on
    # the gates yet - it self-starts as the dual moves. Nothing to assert but
    # the absence of a push.
    assert activation.u.grad.abs().max().item() == 0.0

    # Once the dual has gone negative, it must push the closed gates OPEN:
    # step 1's bias gets a negative gradient, so the optimizer raises it.
    _, opened = dual_gradient(2.0, lambda_raw=-1.0)
    assert opened.u.grad[1].mean().item() < 0

    # Over budget (target 0.5): dual ascends -> lambda > 0, squeezing depth down,
    # and the same gate bias is pushed the other way.
    over, squeezed = dual_gradient(0.5, lambda_raw=1.0)
    assert over < 0
    assert squeezed.u.grad[1].mean().item() > 0


def test_no_zero_dim_parameters():
    """The schedule_free wrapper swaps parameters with
    ``x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))``, which raises
    "self.dim() cannot be 0 to view Float as Byte" on a 0-dim tensor. Every
    parameter must therefore have at least one dimension. This reproduces the
    exact operation rather than just asserting on shape."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)

    modules = {
        "Ouroboros": _built(Ouroboros, x),
        "OuroborosBudget": REGULARIZER_REGISTRY["ouroboros_budget"](),
        "Servant": _built(Servant, x),
    }
    for tag, module in modules.items():
        for name, param in module.named_parameters():
            assert param.dim() > 0, f"{tag}.{name} is 0-dim"
            # The swap itself, byte-for-byte.
            param.detach().clone().view(torch.uint8)


def test_reset_drops_graphs_from_a_labels_free_forward():
    """A training forward without labels never reaches the regularizer, so its
    pushed entries must not survive into the next step.

    If they do, the next drain folds a stale graph into the loss and backward
    dies with "modified by an inplace operation" - the optimizer has swapped
    those parameters in place in between. This reproduces that sequence,
    including the real schedule_free swap.
    """

    def swap(x, y):  # verbatim from pytorch_optimizer's schedulefree
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))
        y.view(torch.uint8).bitwise_xor_(x.view(torch.uint8))
        x.view(torch.uint8).bitwise_xor_(y.view(torch.uint8))

    torch.manual_seed(0)
    x = torch.randn(2, 4, 16)
    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()

    # Step 1: a labels-free forward. The regularizer is NOT called.
    activation(x)

    # The optimizer then swaps every parameter in place, as schedule_free does.
    with torch.no_grad():
        for param in activation.parameters():
            swap(param, torch.zeros_like(param))

    # Step 2: reset (as the model forward now does), then a real labelled step.
    regularizer.reset()
    y = activation(x)
    loss = y.sum() + regularizer(y, torch.zeros(2, 4, dtype=torch.long))
    loss.backward()  # would raise without the reset

    assert all(p.grad is not None for p in activation.parameters())


def test_exit_distribution_is_a_distribution():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)

    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()
    drain_step_counts()

    regularizer(activation(x), torch.zeros(4, 8, dtype=torch.long))
    metrics = regularizer.training_metrics()

    bins = [metrics[f"ouroboros_exit_{k}"] for k in range(MAX_STEPS + 1)]
    assert all(b >= 0.0 for b in bins)
    assert abs(sum(bins) - 1.0) < 1e-5, bins
    # Saturated init: nearly every feature stops after one step.
    assert bins[1] > 0.9

    # Uniform gates across features, so depth is uniform: zero by construction,
    # which is what makes any later spread readable as learned differentiation.
    assert metrics["ouroboros_steps_std"] < 1e-4


def test_spread_detects_a_deep_shallow_split():
    """The mean cannot tell a uniform half-commit from a genuine split. This is
    the measurement the specialization claim actually rests on."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)

    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()

    # Open steps 1 and 2 for half the features: that half runs ~3 steps, the
    # rest still stop at ~1.
    with torch.no_grad():
        activation.u[1, :16] = 6.0
        activation.u[2, :16] = 6.0
    drain_step_counts()  # discard the materialization pass

    regularizer(activation(x), torch.zeros(4, 8, dtype=torch.long))
    metrics = regularizer.training_metrics()

    assert 1.9 < metrics["ouroboros_steps"] < 2.2
    assert metrics["ouroboros_steps_std"] > 0.8
    # Bimodal: mass at 1 step and at 3 steps, not smeared across the middle.
    assert metrics["ouroboros_exit_1"] > 0.4
    assert metrics["ouroboros_exit_3"] > 0.4


def test_token_spread_is_independent_of_feature_spread():
    """Depth can vary across features or across tokens, and they mean different
    things - specialization versus per-token adaptive compute. Averaging over
    tokens before recording made the second invisible, which mattered: read at
    ~2.0 steps, only ~1.5% of the total depth variance was between features.

    Here the gate coupling to token energy is uniform across features and the
    batch carries wildly different per-token energies, so the feature axis must
    read flat while the token axis does not.
    """
    torch.manual_seed(0)
    energies = torch.linspace(0.05, 20.0, 16).view(1, 16, 1)
    x = torch.randn(4, 16, 32) * energies

    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()
    with torch.no_grad():
        # 3.0 sufficed while `m` was a SATURATED tanh, where this 6-nat energy
        # spread mapped to a near-full +/-0.995. Standardizing the signal maps
        # the same spread to +/-0.70 by design - amplitude traded for a live
        # gradient - so a given gate swing costs more coupling. The property
        # under test (token axis moves, feature axis flat) is unchanged.
        activation.p.fill_(5.0)  # same energy coupling for every feature
        activation.u[1].fill_(0.0)  # unsaturate two steps so depth can move
        activation.u[2].fill_(0.0)
    drain_step_counts()

    regularizer(activation(x), torch.zeros(4, 16, dtype=torch.long))
    metrics = regularizer.training_metrics()

    assert metrics["ouroboros_steps_std"] < 1e-4
    assert metrics["ouroboros_token_std"] > 0.5


def test_accounting_drains_and_skips_eval():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)

    regularizer = REGULARIZER_REGISTRY["ouroboros_budget"]()
    activation = _built(Ouroboros, x).train()
    drain_step_counts()

    activation(x)
    activation(x)
    recorded = drain_step_counts()
    assert len(recorded) == 2
    survival, spread = recorded[0]
    assert survival.shape == (MAX_STEPS,)
    assert spread.shape == (9,)
    assert not drain_step_counts(), "drain must clear the stack"

    # Eval must not retain graphs, and an empty stack must not blow up.
    activation.eval()
    activation(x)
    assert not drain_step_counts()
    assert float(regularizer(x, torch.zeros(2, 5, dtype=torch.long))) == 0.0
