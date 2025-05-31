import pytest
import torch
from torch import nn

from praxis.strategies import STRATEGIES_REGISTRY


@pytest.fixture(params=list(STRATEGIES_REGISTRY.keys()))
def strategy_setup(request):
    """Initialize a strategy module for testing."""
    strategy_name = request.param
    strategy_class = STRATEGIES_REGISTRY[strategy_name]

    # Configure the strategy with the right parameters
    if strategy_name == "weighted":
        strategy = strategy_class()
    else:
        strategy = strategy_class()

    return strategy


def test_naive_strategy():
    """Test naive summation strategy."""
    strategy = STRATEGIES_REGISTRY["naive"]()

    # Create sample losses
    loss1 = torch.tensor(2.0, requires_grad=True)
    loss2 = torch.tensor(3.0, requires_grad=True)
    loss3 = torch.tensor(1.5, requires_grad=True)

    losses = [loss1, loss2, loss3]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Verify the combined loss is the sum of the individual losses
    expected_loss = loss1 + loss2 + loss3
    assert torch.isclose(combined_loss, expected_loss)

    # Test that gradients flow correctly
    combined_loss.backward()
    assert loss1.grad is not None
    assert loss2.grad is not None
    assert loss3.grad is not None


def test_uncertainty_weighted_strategy():
    """Test uncertainty weighted strategy."""
    strategy = STRATEGIES_REGISTRY["weighted"]()

    # Create sample losses
    loss1 = torch.tensor(2.0, requires_grad=True)
    loss2 = torch.tensor(3.0, requires_grad=True)
    loss3 = torch.tensor(1.5, requires_grad=True)

    losses = [loss1, loss2, loss3]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Verify that the combined loss is a scalar tensor
    assert combined_loss.dim() == 0
    assert not torch.isnan(combined_loss)
    assert combined_loss.requires_grad

    # Test that parameters were initialized correctly
    assert strategy.params is not None
    assert strategy.params.shape == torch.Size([3])

    # Test that gradients flow correctly
    combined_loss.backward()
    assert loss1.grad is not None
    assert loss2.grad is not None
    assert loss3.grad is not None
    assert strategy.params.grad is not None


def test_strategy_with_imbalanced_losses():
    """Test strategies with imbalanced loss values."""
    for strategy_name, strategy_class in STRATEGIES_REGISTRY.items():
        # Initialize with appropriate parameters for each strategy
        if strategy_name == "weighted":
            strategy = strategy_class()
        else:
            strategy = strategy_class()

        # Create imbalanced losses
        loss1 = torch.tensor(0.01, requires_grad=True)  # Very small loss
        loss2 = torch.tensor(100.0, requires_grad=True)  # Very large loss

        losses = [loss1, loss2]

        # Compute combined loss
        combined_loss = strategy(losses)

        # Verify the combined loss is valid
        assert torch.is_tensor(combined_loss)
        assert not torch.isnan(combined_loss)
        assert combined_loss.requires_grad

        # Test that gradients flow correctly
        combined_loss.backward()
        assert loss1.grad is not None
        assert loss2.grad is not None


def test_strategies_parameterized(strategy_setup):
    """Test all strategies with parameterized fixture."""
    strategy = strategy_setup

    # Create sample losses
    losses = [torch.tensor(1.0, requires_grad=True) for _ in range(3)]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Verify the combined loss is valid
    assert torch.is_tensor(combined_loss)
    assert not torch.isnan(combined_loss)
    assert combined_loss.requires_grad

    # Test that gradients flow correctly
    combined_loss.backward()
    for loss in losses:
        assert loss.grad is not None


def test_real_time_strategy():
    """Test real-time strategy."""
    strategy = STRATEGIES_REGISTRY["real_time"]()

    # Create sample losses
    loss1 = torch.tensor(2.0, requires_grad=True)
    loss2 = torch.tensor(3.0, requires_grad=True)
    loss3 = torch.tensor(1.5, requires_grad=True)

    losses = [loss1, loss2, loss3]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Verify that the combined loss is valid
    assert combined_loss.dim() == 0
    assert not torch.isnan(combined_loss)
    assert combined_loss.requires_grad
    assert torch.allclose(combined_loss, torch.tensor(float(len(losses))))

    # Test that gradients flow correctly
    combined_loss.backward()
    assert loss1.grad is not None
    assert loss2.grad is not None
    assert loss3.grad is not None


def test_uncertainty_weighted_with_negative_losses():
    """Test uncertainty weighted strategy with negative losses (RL rewards)."""
    strategy = STRATEGIES_REGISTRY["weighted"]()

    # Create mixed positive and negative losses
    supervised_loss = torch.tensor(2.0, requires_grad=True)
    rl_reward_loss = torch.tensor(-1.5, requires_grad=True)  # High reward -> negative loss
    auxiliary_loss = torch.tensor(0.8, requires_grad=True)

    losses = [supervised_loss, rl_reward_loss, auxiliary_loss]

    # Compute combined loss multiple times to test stability
    for step in range(5):
        combined_loss = strategy(losses)
        
        # Verify the combined loss is valid
        assert combined_loss.dim() == 0
        assert not torch.isnan(combined_loss)
        assert torch.isfinite(combined_loss)
        assert combined_loss.requires_grad
        
        # Test gradient flow
        combined_loss.backward(retain_graph=True)
        assert supervised_loss.grad is not None
        assert rl_reward_loss.grad is not None
        assert auxiliary_loss.grad is not None
        assert strategy.params.grad is not None
        
        # Clear gradients for next iteration
        supervised_loss.grad.zero_()
        rl_reward_loss.grad.zero_()
        auxiliary_loss.grad.zero_()
        strategy.params.grad.zero_()
        
    # Verify parameters remain stable (no extreme values)
    assert torch.all(torch.isfinite(strategy.params))
    assert torch.all(torch.abs(strategy.params) < 100)  # Reasonable bounds


def test_uncertainty_weighted_all_negative_losses():
    """Test uncertainty weighted strategy with all negative losses (all rewards)."""
    strategy = STRATEGIES_REGISTRY["weighted"]()

    # All negative losses (representing reward signals)
    reward_loss1 = torch.tensor(-2.0, requires_grad=True)
    reward_loss2 = torch.tensor(-1.5, requires_grad=True)
    reward_loss3 = torch.tensor(-0.8, requires_grad=True)

    losses = [reward_loss1, reward_loss2, reward_loss3]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Should be negative (since we want to maximize all rewards)
    assert combined_loss.item() < 0
    assert not torch.isnan(combined_loss)
    assert torch.isfinite(combined_loss)
    assert combined_loss.requires_grad

    # Test gradient flow
    combined_loss.backward()
    assert reward_loss1.grad is not None
    assert reward_loss2.grad is not None  
    assert reward_loss3.grad is not None
    assert strategy.params.grad is not None


def test_uncertainty_weighted_extreme_values():
    """Test uncertainty weighted strategy with extreme positive and negative values."""
    strategy = STRATEGIES_REGISTRY["weighted"]()

    # Extreme values
    large_positive_loss = torch.tensor(100.0, requires_grad=True)
    large_negative_loss = torch.tensor(-50.0, requires_grad=True)
    small_positive_loss = torch.tensor(0.001, requires_grad=True)
    small_negative_loss = torch.tensor(-0.0001, requires_grad=True)

    losses = [large_positive_loss, large_negative_loss, small_positive_loss, small_negative_loss]

    # Compute combined loss
    combined_loss = strategy(losses)

    # Verify numerical stability
    assert not torch.isnan(combined_loss)
    assert torch.isfinite(combined_loss)
    assert combined_loss.requires_grad

    # Test gradient flow
    combined_loss.backward()
    for loss in losses:
        assert loss.grad is not None
        assert torch.isfinite(loss.grad)
    
    assert strategy.params.grad is not None
    assert torch.all(torch.isfinite(strategy.params.grad))


def test_uncertainty_weighted_sign_preservation():
    """Test that uncertainty weighted strategy preserves loss sign correctly."""
    strategy = STRATEGIES_REGISTRY["weighted"]()

    # Positive loss should contribute positively to total loss
    positive_loss = torch.tensor(1.0, requires_grad=True)
    positive_only_losses = [positive_loss]
    positive_combined = strategy(positive_only_losses)
    
    # Reset strategy for fair comparison
    strategy = STRATEGIES_REGISTRY["weighted"]()
    
    # Negative loss should contribute negatively to total loss (reward maximization)
    negative_loss = torch.tensor(-1.0, requires_grad=True)
    negative_only_losses = [negative_loss]
    negative_combined = strategy(negative_only_losses)
    
    # The negative loss case should result in a more negative total
    # (accounting for regularization terms)
    assert negative_combined.item() < positive_combined.item()
