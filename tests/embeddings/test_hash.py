"""HashEmbedding's bucket codes stay independent across its hash functions,
even at a power-of-two bucket count."""

import torch

from praxis.embeddings.hash import byte_group_hash_function, rolling_polynomial_hash


def _distinct(codes):
    return len(set(map(tuple, codes.tolist())))


def test_four_hashes_stay_independent_at_a_power_of_two():
    letters = torch.tensor(list(b"etaoinshrdlucmf "))
    grams = torch.cartesian_prod(*[letters] * 4)  # 65536 distinct 4-grams
    mixed = torch.stack(
        [byte_group_hash_function(grams, 4, f, 256)[:, -1] for f in range(4)], dim=1
    )
    assert _distinct(mixed) > 0.999 * len(grams)
    # Unmixed, the polynomial keeps only low bits that every odd prime shapes alike.
    raw = torch.stack(
        [rolling_polynomial_hash(grams, f) % 256 for f in range(4)], dim=1
    )
    assert _distinct(raw) < 0.9 * len(grams)
