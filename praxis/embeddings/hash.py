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

    Collision resistance comes from ``functions``, not from ``hash_vocab``. A
    single hash maps every distinct n-gram onto exactly one row, so two n-grams
    sharing a bucket become the same vector and are unrecoverable downstream;
    the only defence is a wider table, and byte n-grams outnumber any affordable
    table. ``M`` independent hashes make an n-gram's code the M-tuple of buckets,
    which is ambiguous only when all M collide at once. Measured on 8MB of
    minipile, 5-byte windows: M=1 over 4096 buckets leaves 100% of occurrences
    ambiguous, while M=4 over 1024 buckets - the same row count, so the same
    parameters - leaves under 2%. Widening a single table cannot buy this; 65536
    buckets at M=1 still leaves ~97% ambiguous at 16x the parameters.
    """

    def __init__(
        self, config, encoder=None, group_sizes=None, functions=1, hash_vocab=None
    ):
        super().__init__()
        # Dim is architectural (from the encoder/config); the n-gram sizes,
        # function count and bucket count come from the embedding profile.
        self.dim = encoder.input_dim if encoder is not None else config.embed_size
        self.group_sizes = group_sizes if group_sizes is not None else [3, 4, 5]
        self.nb_functions = functions
        # Buckets per table, scalar or one entry per group size, and read from
        # `config.hash_buckets` (the --hash-buckets flag) so the number that
        # decides this table's size is visible in the experiment file rather
        # than buried in a registry profile. Per-group counts exist because the
        # number of distinct n-grams in real text grows roughly fourfold per
        # added byte, so one scalar either starves the long windows or wastes
        # rows on the short ones.
        #
        # Unset falls back to vocab_size, which is what every config predating
        # the flag was doing. That fallback is legacy, not intent: this table
        # indexes byte n-grams and has nothing to do with a vocabulary.
        if hash_vocab is None:
            hash_vocab = getattr(config, "hash_buckets", None)
        if hash_vocab is None:
            hash_vocab = config.vocab_size
        if isinstance(hash_vocab, int):
            hash_vocab = [hash_vocab]
        self.hash_vocab = list(hash_vocab)
        if len(self.hash_vocab) == 1:
            self.hash_vocab *= len(self.group_sizes)
        if len(self.hash_vocab) != len(self.group_sizes):
            raise ValueError(
                f"hash_buckets has {len(self.hash_vocab)} entries but there are "
                f"{len(self.group_sizes)} n-gram window sizes "
                f"{self.group_sizes}; pass one value or one per window"
            )
        if self.nb_functions > len(PRIMES):
            raise ValueError(
                f"HashEmbedding supports at most {len(PRIMES)} hash functions "
                f"(one per prime), got {self.nb_functions}"
            )

        self.embeddings = nn.ModuleList(
            nn.Embedding(buckets, self.dim)
            for _ in range(self.nb_functions)
            for buckets in self.hash_vocab
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
            for group_size, buckets in zip(self.group_sizes, self.hash_vocab):
                hash_ids = byte_group_hash_function(
                    tokens,
                    group_size=group_size,
                    hash_func_nb=func_nb,
                    max_hash=buckets,
                )
                result = result + self.embeddings[idx](hash_ids)
                idx += 1
        return result

    def __repr__(self) -> str:
        buckets = (
            self.hash_vocab[0]
            if len(set(self.hash_vocab)) == 1
            else list(self.hash_vocab)
        )
        return (
            f"{self.__class__.__name__}("
            f"vocab={buckets}, dim={self.dim}, "
            f"groups={self.group_sizes}, functions={self.nb_functions})"
        )
