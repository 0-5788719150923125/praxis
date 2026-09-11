"""Tests for praxis/embeddings/composite.py (AdditiveEmbedding)."""

from praxis import registry


def test_repr_lists_children_the_way_pytorch_does(embedding_config):
    """It used to join the children with ``+`` on a single line.

    A container's children ARE its repr: ``print(model)`` indents them
    under numbered keys, and every nested module gets to describe itself.
    Flattening them into an expression put the tree on one line and hid any
    structure the children had of their own.
    """
    composed = registry.lookup("embeddings", "byte_hash")(embedding_config())
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
