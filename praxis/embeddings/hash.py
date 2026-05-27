import torch
import torch.nn as nn

# Primes for the polynomial rolling hash (from the BLT reference).
PRIMES = [
    1000000007,
    5915587277,
    1500450271,
    3267000013,
    5754853343,
    4093082899,
    9576890767,
    3628273133,
    2860486313,
    5463458053,
]


def rolling_polynomial_hash(t: torch.Tensor, hash_func_nb: int = 0) -> torch.Tensor:
    """Polynomial rolling hash over the last dim of a windowed tensor."""
    prime = torch.tensor(PRIMES[hash_func_nb], dtype=torch.int64, device=t.device)
    prime_powers = torch.stack([prime**i for i in range(t.shape[-1])])
    return torch.sum(t * prime_powers, dim=-1)


def byte_group_hash_function(
    x: torch.Tensor, group_size: int = 2, hash_func_nb: int = 0, max_hash: int = 30000
) -> torch.Tensor:
    """Hash each length-``group_size`` byte window to a bucket in ``[0, max_hash)``."""
    with torch.no_grad():
        bs, _ = x.shape
        prefix = torch.zeros(bs, group_size - 1, dtype=torch.int64, device=x.device)
        x_padded = torch.cat([prefix, x], dim=1)
        windows = x_padded.unfold(1, group_size, 1)
        hashes = rolling_polynomial_hash(windows, hash_func_nb)
        hash_values_range = hashes % max_hash
    hash_values_range.requires_grad = False
    return hash_values_range


class HashEmbedding(nn.Module):
    """N-gram hash embedding: sums table lookups over byte windows of several
    sizes and hash functions, computing vectors from byte n-grams rather than
    retrieving a per-token row. Has no single tie-able table by design.
    """

    def __init__(self, config, encoder=None, group_sizes=None, functions=1):
        super().__init__()
        # Dim is architectural (from the encoder/config); the n-gram sizes and
        # function count come from the embedding profile.
        self.dim = encoder.input_dim if encoder is not None else config.embed_size
        self.group_sizes = group_sizes if group_sizes is not None else [3, 4, 5]
        self.nb_functions = functions
        # Hash space spans the external vocab so collisions stay rare.
        self.hash_vocab = config.vocab_size

        self.embeddings = nn.ModuleList(
            nn.Embedding(self.hash_vocab, self.dim)
            for _ in range(self.nb_functions)
            for _ in self.group_sizes
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(
            tokens.shape[0],
            tokens.shape[1],
            self.dim,
            device=tokens.device,
            dtype=torch.float32,
        )
        idx = 0
        for func_nb in range(self.nb_functions):
            for group_size in self.group_sizes:
                hash_ids = byte_group_hash_function(
                    tokens,
                    group_size=group_size,
                    hash_func_nb=func_nb,
                    max_hash=self.hash_vocab,
                )
                result = result + self.embeddings[idx](hash_ids)
                idx += 1
        return result

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"vocab={self.hash_vocab}, dim={self.dim}, "
            f"groups={self.group_sizes}, functions={self.nb_functions})"
        )
