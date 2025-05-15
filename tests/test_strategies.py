import pytest
import torch
from torch import nn

from praxis.strategies import STRATEGIES_REGISTRY
from praxis.strategies.naive import NaiveSummation
from praxis.strategies.uncertainty_weighted import UncertaintyWeighted


@pytest.fixture(params=list(STRATEGIES_REGISTRY.values()))
def strategy_setup(request):
    """Initialize a strategy module for testing."""
    strategy = request.param()
    return strategy


def test_naive_strategy():
    """Test naive summation strategy."""
    strategy = NaiveSummation()
    
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
    strategy = UncertaintyWeighted()
    
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
    for strategy_class in STRATEGIES_REGISTRY.values():
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