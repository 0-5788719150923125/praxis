import random

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from builders import (
    HUGGINGFACE_DATASETS,
    HuggingfaceDataset,
    InterleaveDataManager,
    text_formatter,
)
from praxis.controllers import CONTROLLER_REGISTRY
from praxis.controllers.base import BaseController


class MockConfig:
    """Mock configuration for testing controllers."""

    def __init__(self):
        self.debug = False
        self.causal = True
        self.hidden_size = 256
        self.dropout = 0.1
        self.depth = 3
        self.num_experts = 3
        self.num_heads = 4
        self.activation = "swish"


def test_random_seed_determinism():
    """Test that Python's random module produces deterministic results with a fixed seed."""
    # First run with seed 42
    random.seed(42)
    values_1 = [random.random() for _ in range(10)]

    # Second run with same seed
    random.seed(42)
    values_2 = [random.random() for _ in range(10)]

    # Values should be identical
    assert values_1 == values_2, "Random values with the same seed should be identical"

    # Different seed should produce different values
    random.seed(43)
    values_3 = [random.random() for _ in range(10)]
    assert values_1 != values_3, "Random values with different seeds should differ"

    # Using random.choice() for deterministic sequence selection
    items = ["a", "b", "c", "d", "e"]

    random.seed(42)
    choices_1 = [random.choice(items) for _ in range(10)]

    random.seed(42)
    choices_2 = [random.choice(items) for _ in range(10)]

    # Choices should be identical
    assert (
        choices_1 == choices_2
    ), "Random choices should be deterministic with the same seed"


def test_numpy_seed_determinism():
    """Test that NumPy's random module produces deterministic results with a fixed seed."""
    # First run with seed 42
    np.random.seed(42)
    values_1 = np.random.rand(10)

    # Second run with same seed
    np.random.seed(42)
    values_2 = np.random.rand(10)

    # Values should be identical
    np.testing.assert_array_equal(values_1, values_2)

    # Different seed should produce different values
    np.random.seed(43)
    values_3 = np.random.rand(10)
    assert not np.array_equal(values_1, values_3)

    # Test deterministic sampling of indices (used in dataset)
    np.random.seed(42)
    indices_1 = np.random.choice(100, 10, replace=False)

    np.random.seed(42)
    indices_2 = np.random.choice(100, 10, replace=False)

    # Indices should be identical
    np.testing.assert_array_equal(indices_1, indices_2)


def test_torch_seed_determinism():
    """Test that PyTorch produces deterministic results with a fixed seed."""
    # First run with seed 42
    torch.manual_seed(42)
    values_1 = torch.rand(10)

    # Second run with same seed
    torch.manual_seed(42)
    values_2 = torch.rand(10)

    # Values should be identical
    torch.testing.assert_close(values_1, values_2)

    # Different seed should produce different values
    torch.manual_seed(43)
    values_3 = torch.rand(10)
    assert not torch.allclose(values_1, values_3)


def test_complete_torch_determinism():
    """Test comprehensive PyTorch determinism settings."""

    def get_random_values():
        # Create some random operations
        a = torch.rand(5, 5)
        b = torch.rand(5, 5)
        c = torch.mm(a, b)
        return a, b, c

    # Setup fully deterministic behavior
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enable deterministic algorithms with warning only (some ops might not have deterministic implementations)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # First run
    a1, b1, c1 = get_random_values()

    # Reset all seeds and run again
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    a2, b2, c2 = get_random_values()

    # Check equality
    torch.testing.assert_close(a1, a2)
    torch.testing.assert_close(b1, b2)
    torch.testing.assert_close(c1, c2)


def test_base_controller_determinism():
    """Test that BaseController routing is deterministic."""
    config = MockConfig()
    controller = BaseController(config)

    # Create test inputs
    batch_size = 2
    seq_len = 5
    hidden_dim = 256

    # Set all random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hidden_states = torch.rand(batch_size, seq_len, hidden_dim)

    # Create mock experts
    sequential_experts = [
        torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(config.num_experts)
    ]
    ordered_experts = sequential_experts.copy()

    # First run
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    controller_state = None
    current_route_1 = []
    current_depth = 0

    for _ in range(5):
        hidden_states, controller_state, aux_loss, next_expert_idx = (
            controller.get_next_expert(
                hidden_states,
                controller_state,
                sequential_experts,
                ordered_experts,
                current_route_1,
                current_depth,
            )
        )
        if next_expert_idx is not None:
            current_route_1 = controller.update_route(
                hidden_states, current_route_1, current_depth, next_expert_idx
            )
            current_depth = next_expert_idx
        else:
            break

    # Second run - should be identical
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hidden_states = torch.rand(batch_size, seq_len, hidden_dim)  # Reset hidden states
    controller_state = None
    current_route_2 = []
    current_depth = 0

    for _ in range(5):
        hidden_states, controller_state, aux_loss, next_expert_idx = (
            controller.get_next_expert(
                hidden_states,
                controller_state,
                sequential_experts,
                ordered_experts,
                current_route_2,
                current_depth,
            )
        )
        if next_expert_idx is not None:
            current_route_2 = controller.update_route(
                hidden_states, current_route_2, current_depth, next_expert_idx
            )
            current_depth = next_expert_idx
        else:
            break

    assert (
        current_route_1 == current_route_2
    ), "Controller routes should be deterministic with the same seed"


@pytest.mark.parametrize(
    "controller_name",
    [
        "base",
        "layer_shuffle",
        "attention",
        "counter_attention",
        "neural",
    ],  # Only test controllers that are deterministic
)
def test_controller_determinism(controller_name):
    """Test that all controllers produce deterministic routes with fixed seeds."""
    config = MockConfig()
    controller_class = CONTROLLER_REGISTRY[controller_name]

    # First run
    torch.manual_seed(42)
    controller_1 = controller_class(config)

    # Create test inputs
    batch_size = 2
    seq_len = 5
    hidden_dim = 256
    hidden_states_1 = torch.rand(batch_size, seq_len, hidden_dim)

    # Create mock experts
    sequential_experts = [
        torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(config.num_experts)
    ]
    ordered_experts = sequential_experts.copy()

    # First routing path
    controller_state = None
    current_route_1 = []
    current_depth = 0

    for _ in range(5):
        hidden_states_1, controller_state, aux_loss, next_expert_idx = (
            controller_1.get_next_expert(
                hidden_states_1,
                controller_state,
                sequential_experts,
                ordered_experts,
                current_route_1,
                current_depth,
            )
        )
        if next_expert_idx is not None:
            current_route_1 = controller_1.update_route(
                hidden_states_1, current_route_1, current_depth, next_expert_idx
            )
            current_depth = next_expert_idx
        else:
            break

    # Second run with same seed - should be identical
    torch.manual_seed(42)
    controller_2 = controller_class(config)
    hidden_states_2 = torch.rand(batch_size, seq_len, hidden_dim)

    controller_state = None
    current_route_2 = []
    current_depth = 0

    for _ in range(5):
        hidden_states_2, controller_state, aux_loss, next_expert_idx = (
            controller_2.get_next_expert(
                hidden_states_2,
                controller_state,
                sequential_experts,
                ordered_experts,
                current_route_2,
                current_depth,
            )
        )
        if next_expert_idx is not None:
            current_route_2 = controller_2.update_route(
                hidden_states_2, current_route_2, current_depth, next_expert_idx
            )
            current_depth = next_expert_idx
        else:
            break

    assert (
        current_route_1 == current_route_2
    ), f"{controller_name} controller routes should be deterministic with the same seed"


def test_huggingface_dataset_determinism():
    """Test that HuggingfaceDataset produces deterministic sequences with the same seed."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Choose a small dataset from the available ones
    dataset_config = HUGGINGFACE_DATASETS["minipile-validation"].copy()
    dataset_config["streaming"] = False  # Set streaming to False for testing

    # Set all random sources before first run
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # First run with seed 42
    dataset_1 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    sequences_1 = dataset_1.get_sequences(5)

    # Reset all random sources before second run
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Second run with same seed
    dataset_2 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    sequences_2 = dataset_2.get_sequences(5)

    # Sequences should be identical
    assert (
        sequences_1 == sequences_2
    ), "Dataset sequences should be deterministic with the same seed"

    # Set different seed for third run
    random.seed(43)
    torch.manual_seed(43)
    np.random.seed(43)

    # Different seed should give different sequences
    dataset_3 = HuggingfaceDataset(tokenizer, 43, dataset_config)
    sequences_3 = dataset_3.get_sequences(5)

    # Unlikely that all sequences would match with a different seed
    assert (
        sequences_1 != sequences_3
    ), "Dataset sequences should differ with different seeds"


def test_interleave_data_manager_determinism():
    """Test that InterleaveDataManager produces deterministic batches with fixed seeds."""
    # Create a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    # Use a small dataset
    dataset_config = HUGGINGFACE_DATASETS["minipile-validation"].copy()
    dataset_config["streaming"] = False

    # Create two datasets with the same seed - set all random sources
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    dataset_1 = HuggingfaceDataset(tokenizer, 42, dataset_config)
    dataset_2 = HuggingfaceDataset(tokenizer, 42, dataset_config)

    # Create two data managers with the same settings
    manager_1 = InterleaveDataManager([dataset_1], [1.0], tokenizer, block_size=128)
    manager_2 = InterleaveDataManager([dataset_2], [1.0], tokenizer, block_size=128)

    # Get batches from both managers
    batch_1 = manager_1.get_batch(batch_size=2)

    # Reset all seeds
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    batch_2 = manager_2.get_batch(batch_size=2)

    # Check that batches are identical
    for i in range(len(batch_1)):
        torch.testing.assert_close(batch_1[i], batch_2[i])

    # Test different sampling modes
    batch_3 = manager_1.get_batch(batch_size=4, oversample=True)

    # Reset all seeds and get new batch
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    batch_4 = manager_2.get_batch(batch_size=4, oversample=True)

    # Check that batches are identical
    for i in range(len(batch_3)):
        torch.testing.assert_close(batch_3[i], batch_4[i])


def test_text_formatter_determinism():
    """Test that text_formatter function is deterministic (should always be since it's rule-based)."""
    test_inputs = [
        "First paragraph.\nSecond paragraph.",
        "Code block:\ndef test():\n    print('hello')",
        "List items:\n- Item 1\n- Item 2\n- Item 3",
        "Multiple\n\n\nnewlines.",
    ]

    # Process each input twice
    for text in test_inputs:
        result_1 = text_formatter(text)
        result_2 = text_formatter(text)

        # Results should be identical
        assert (
            result_1 == result_2
        ), "text_formatter should always return the same result for the same input"
