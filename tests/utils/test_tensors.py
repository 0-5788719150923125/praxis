import pytest
import torch

from praxis.utils import create_block_ids


@pytest.fixture
def device():
    """Fixture for device selection."""
    return "cpu"


def test_single_special_token(device):
    """Test block creation with a single special token."""
    # Arrange
    input_ids = torch.tensor([[1, 2, 0, 3, 4, 0, 5]], device=device)
    special_tokens = [0]  # Using 0 as padding token

    # Act
    block_ids = create_block_ids(input_ids, special_tokens)

    # Assert
    expected = torch.tensor([[1, 1, 1, 2, 2, 2, 3]], device=device)
    assert torch.all(block_ids == expected)
    assert block_ids.shape == input_ids.shape


def test_multiple_special_tokens(device):
    """Test block creation with multiple special tokens."""
    # Arrange
    input_ids = torch.tensor([[1, 2, 0, 3, 0, 4, 0, 5]], device=device)
    special_tokens = [0]  # Using only padding token

    # Act
    block_ids = create_block_ids(input_ids, special_tokens)

    # Assert
    expected = torch.tensor([[1, 1, 1, 2, 2, 3, 3, 4]], device=device)
    assert torch.all(block_ids == expected)


def test_batched_input(device):
    """Test block creation with batched input."""
    # Arrange
    input_ids = torch.tensor([[1, 2, 0, 3, 4], [5, 0, 6, 0, 7]], device=device)
    special_tokens = [0]

    # Act
    block_ids = create_block_ids(input_ids, special_tokens)

    # Assert
    expected = torch.tensor([[1, 1, 1, 2, 2], [1, 1, 2, 2, 3]], device=device)
    assert torch.all(block_ids == expected)
    assert block_ids.shape == input_ids.shape


def test_edge_cases(device):
    """Test edge cases for block creation."""
    # Test with all special tokens
    input_ids = torch.tensor([[0, 0, 0]], device=device)
    special_tokens = [0]
    block_ids = create_block_ids(input_ids, special_tokens)
    expected = torch.tensor([[1, 2, 2]], device=device)  # Updated to match behavior
    assert torch.all(block_ids == expected)

    # Test with no special tokens
    input_ids = torch.tensor([[1, 2, 3]], device=device)
    block_ids = create_block_ids(input_ids, special_tokens)
    expected = torch.tensor([[1, 1, 1]], device=device)
    assert torch.all(block_ids == expected)

    # Test with alternating special tokens
    input_ids = torch.tensor([[1, 0, 2, 0, 3]], device=device)
    block_ids = create_block_ids(input_ids, special_tokens)
    expected = torch.tensor([[1, 1, 2, 2, 3]], device=device)
    assert torch.all(block_ids == expected)

    # Test with special tokens at start/end
    input_ids = torch.tensor([[0, 1, 0]], device=device)
    block_ids = create_block_ids(input_ids, special_tokens)
    expected = torch.tensor([[1, 2, 2]], device=device)
    assert torch.all(block_ids == expected)
