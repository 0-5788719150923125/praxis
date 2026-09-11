"""Tests for praxis.embeddings module."""

from dataclasses import dataclass

import pytest
import torch.nn as nn

from praxis import registry
from praxis.embeddings import AdditiveEmbedding
from praxis.embeddings.byte import ByteEmbedding
from praxis.embeddings.hash import HashEmbedding
from praxis.embeddings.positional import PositionalEmbedding
from praxis.embeddings.projected import ProjectedEmbedding


@dataclass
class MockConfig:
    """Mock configuration for testing embeddings."""

    vocab_size: int = 1000
    embed_size: int = 64
    hidden_size: int = 128
    max_position_embeddings: int = 512
    dropout: float = 0.1


class TestEmbeddingRegistry:
    """Test the embedding registry."""

    def test_registry_contains_expected_architectures(self):
        """Test that all expected architectures map to the right class."""
        expected = {
            "conv": ProjectedEmbedding,
            "min": ProjectedEmbedding,
            "mru": PositionalEmbedding,
            "nano": ProjectedEmbedding,
            "recurrent": ProjectedEmbedding,
            "transformer": ProjectedEmbedding,
            # Byte-latent primitives, composed by profiles.
            "tok": ByteEmbedding,
            "hash": HashEmbedding,
        }

        for arch, cls in expected.items():
            assert arch in registry.namespace("embeddings")
            assert registry.lookup("embeddings", arch) == cls

    def test_registry_values_are_callables(self):
        """Registry values are callable; class entries are nn.Module subclasses."""
        for arch, ctor in registry.namespace("embeddings").items():
            assert callable(ctor)
            if isinstance(ctor, type):
                assert issubclass(ctor, nn.Module)

    def test_byte_latent_profiles_build(self):
        """The byte-latent profiles compose the expected primitives."""
        config = MockConfig()

        tok_only = registry.lookup("embeddings", "byte")(config)
        assert isinstance(tok_only, ByteEmbedding)

        tok_hash = registry.lookup("embeddings", "byte_hash")(config)
        assert isinstance(tok_hash, AdditiveEmbedding)
        kinds = [type(m) for m in tok_hash.embeddings]
        assert kinds == [ByteEmbedding, HashEmbedding]
        # The byte table is the tie source; the hash branch has no single weight.
        assert tok_hash.tie_source() is tok_hash.embeddings[0]

    def test_repr_lists_children_the_way_pytorch_does(self):
        """It used to join the children with ``+`` on a single line.

        A container's children ARE its repr: ``print(model)`` indents them
        under numbered keys, and every nested module gets to describe itself.
        Flattening them into an expression put the tree on one line and hid any
        structure the children had of their own.
        """
        composed = registry.lookup("embeddings", "byte_hash")(MockConfig())
        lines = repr(composed).splitlines()

        assert " + " not in repr(composed)
        assert lines[0] == "AdditiveEmbedding("
        assert lines[-1] == ")"
        # One indented, numbered entry per primitive, under the ModuleList.
        assert "(embeddings): ModuleList(" in lines[1]
        for index, module in enumerate(composed.embeddings):
            assert any(
                line.strip().startswith(f"({index}): {type(module).__name__}(")
                for line in lines
            ), f"child {index} is not listed"
