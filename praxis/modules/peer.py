import math
from typing import Dict, List, Optional, Tuple, TypeVar, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.activations import ACT2FN

ConfigType = TypeVar('ConfigType', bound='AutoConfig')


class PraxisPEER(nn.Module):
    """
    This class implements the Parameter-Efficient Expert Retrieval (PEER) mechanism:
    https://arxiv.org/abs/2407.04153v1
    
    PEER combines aspects of product key memory and mixture of experts, 
    using factorized keys for efficient expert retrieval. It enables each token
    to select its own set of experts for processing.
    """

    __version__ = "0.1.0"

    def __init__(self, config: ConfigType):
        """
        Initialize the PEER module.
        
        Args:
            config: Configuration object containing PEER parameters
        """
        super().__init__()

        hidden_size = config.hidden_size
        key_dims = config.expert["key_dims"]
        self.k: int = config.expert["k"]
        self.num_heads: int = config.expert["num_heads"]
        self.offset_heads: bool = config.expert["offset_heads"]
        self.num_experts: int = config.expert["num_experts"]
        self.num_sets: int = 1 if not self.offset_heads else self.num_heads

        # Product-Key retrieval requires keys to be a perfect square of the total experts
        self.num_keys: int = int(math.sqrt(self.num_experts))

        assert (
            self.num_experts**0.5
        ).is_integer(), "`self.num_experts` needs to be a perfect square"
        assert (hidden_size % 2) == 0, "`hidden_size` should be divisible by 2"

        class Permute(nn.Module):
            """Permute dimensions of tensor for product key memory."""
            
            def __init__(self):
                super().__init__()

            def forward(self, x: Tensor) -> Tensor:
                """
                Permute dimensions [p, b, n, h, d] → [p, b, n, h, d]
                
                Args:
                    x: Input tensor
                    
                Returns:
                    Permuted tensor
                """
                return x.permute(2, 0, 1, 3, 4).contiguous()

        # BatchNorm for combined partitions and heads
        class BatchNorm1d(nn.BatchNorm1d):
            """BatchNorm1d that handles sequence dimension."""
            
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)

            def forward(self, x: Tensor) -> Tensor:
                """
                Apply batch norm across batch and sequence dimensions.
                
                Args:
                    x: Input tensor of shape [batch_size, seq_len, dim]
                    
                Returns:
                    Normalized tensor of same shape
                """
                b, s, d = x.shape
                x = x.view(b * s, d)
                x = super().forward(x)
                return x.view(b, s, d)

        self.queries = nn.Sequential(
            BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, key_dims * self.num_heads * 2, bias=False),
            nn.Unflatten(-1, (2, self.num_heads, key_dims)),
            Permute(),
        )

        self.keys = nn.Parameter(
            torch.randn(self.num_heads, self.num_keys, 2, key_dims)
        )
        nn.init.normal_(self.keys, std=0.02)

        self.down = nn.Embedding(self.num_experts * self.num_sets, hidden_size)
        nn.init.xavier_uniform_(self.down.weight)
        self.act = ACT2FN[config.activation]
        self.dropout = nn.Dropout(config.dropout)
        self.up = nn.Embedding(self.num_experts * self.num_sets, hidden_size)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the PEER module.
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Generate queries
        queries = self.queries(
            inputs
        )  # Shape: (2, batch_size, seq_len, heads, dim_key)

        # Compute similarities using Einstein summation
        sim = torch.einsum("p b n h d, h k p d -> p b n h k", queries, self.keys)

        # For each partition, get top-k indices and scores
        scores_parts, indices_parts = sim.topk(self.k, dim=-1)
        scores_x, scores_y = scores_parts
        indices_x, indices_y = indices_parts

        # Compute Cartesian product of top-k indices and scores
        all_scores = scores_x.unsqueeze(-1) + scores_y.unsqueeze(-2)
        all_indices = indices_x.unsqueeze(-1) * self.num_keys + indices_y.unsqueeze(-2)

        # Flatten last two dimensions
        all_scores = all_scores.view(
            *all_scores.shape[:-2], math.prod(all_scores.shape[-2:])
        )
        all_indices = all_indices.view(
            *all_indices.shape[:-2], math.prod(all_indices.shape[-2:])
        )

        # Get top expert keys from the Cartesian product
        scores, pk_indices = all_scores.topk(self.k, dim=-1)
        indices = all_indices.gather(-1, pk_indices)

        if self.offset_heads:
            head_expert_offsets = (
                torch.arange(self.num_heads, device=inputs.device) * self.num_experts
            )
            indices = indices + head_expert_offsets.view(1, 1, -1, 1)

        # Lookup expert weights using embeddings
        weights_down = self.down(indices)
        outputs = torch.einsum("b n d, b n h k d -> b n h k", inputs, weights_down)

        # Apply sigmoid scores to activated outputs
        outputs = F.sigmoid(scores) * self.act(outputs)

        # Force sparse ensembling of intermediate states
        outputs = self.dropout(outputs)
        weights_up = self.dropout(self.up(indices))

        # Aggregate expert outputs
        outputs = torch.einsum("b n h k, b n h k d -> b n d", outputs, weights_up)

        return outputs


if __name__ == "__main__":
    # Define a suite of configurations to test
    # Note: We ensure num_experts is always a perfect square for correctness.
    test_configs = [
        {
            "hidden_size": 256,
            "activation": "gelu",
            "dropout": 0.1,
            "expert": {
                "key_dims": 16,
                "k": 8,
                "num_heads": 4,
                "offset_heads": False,
                "num_experts": 64,  # 8x8 = 64
            },
            "description": "Base configuration",
        },
        {
            "hidden_size": 512,
            "activation": "gelu",
            "dropout": 0.1,
            "expert": {
                "key_dims": 32,
                "k": 16,
                "num_heads": 8,
                "offset_heads": False,
                "num_experts": 256,  # 16x16 = 256
            },
            "description": "Larger hidden size, more experts, bigger k",
        },
        {
            "hidden_size": 256,
            "activation": "gelu",
            "dropout": 0.1,
            "expert": {
                "key_dims": 16,
                "k": 4,
                "num_heads": 2,
                "offset_heads": True,
                "num_experts": 49,  # 7x7 = 49
            },
            "description": "Fewer heads, smaller k, offset heads enabled",
        },
        {
            "hidden_size": 128,
            "activation": "gelu",
            "dropout": 0.1,
            "expert": {
                "key_dims": 8,
                "k": 8,
                "num_heads": 4,
                "offset_heads": False,
                "num_experts": 81,  # 9x9=81
            },
            "description": "Smaller hidden size, moderate experts",
        },
    ]

    # Define a set of test input shapes (batch_size, seq_len)
    test_input_shapes = [
        (2, 16),
        (8, 64),
        (16, 128),
        (32, 128),
    ]

    # Helper function for running tests on a given model and input
    def run_test(model, batch_size, seq_len, hidden_size):
        inp = torch.randn(batch_size, seq_len, hidden_size, device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()
        out = model(inp)
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated()
        mem_diff_mb = (mem_after - mem_before) / (1024**2)
        return out.shape, mem_diff_mb

    # Iterate over each config
    for cfg in test_configs:
        # Dynamically build a config object
        class Config:
            pass

        config = Config()
        config.hidden_size = cfg["hidden_size"]
        config.activation = cfg["activation"]
        config.dropout = cfg["dropout"]
        config.expert = cfg["expert"]

        print(f"=== Testing Configuration: {cfg['description']} ===")
        print(config.expert)
        print(
            f"hidden_size={config.hidden_size}, activation={config.activation}, dropout={config.dropout}"
        )

        model = PraxisPEER(config)
        model = model.to("cuda")

        # Test multiple input shapes
        for bs, sl in test_input_shapes:
            shape, mem_usage = run_test(model, bs, sl, config.hidden_size)
            print(
                f"Input: batch_size={bs}, seq_len={sl} => Output Shape: {shape}, Memory Diff: {mem_usage:.2f} MB"
            )

        print("---------------------------------------------------")
