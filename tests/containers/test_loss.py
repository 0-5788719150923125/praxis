"""LossContainer: named loss terms that accumulate per key and merge across containers."""

import pytest
import torch

from praxis.containers import LossContainer


def test_container_names_and_values_stay_aligned():
    c = LossContainer()
    c.add_loss("main", torch.tensor(1.0))
    c.add_loss("mtp", torch.tensor(2.0))
    names, values = c.get_named_losses()
    assert names == ["main", "mtp"]
    assert [v.item() for v in values] == [1.0, 2.0]
    assert values == c.get_loss_values()


def test_add_loss_accumulates_on_a_repeated_key_and_wraps_numbers():
    c = LossContainer(router=0.25)
    c.add_loss("router", 0.5)
    c.add_loss("router", 1)
    c.add_loss("unknown", "not a loss")  # non-numeric values count as zero

    assert set(c.loss_dict) == {"main", "router", "unknown"}
    for key in c.loss_dict:
        assert isinstance(c.get_loss(key), torch.Tensor)
    assert c.get_loss("main").item() == 0.0
    assert c.get_loss("router").item() == pytest.approx(1.75)
    assert c.get_loss("unknown").item() == 0.0
    assert "router" in c and "missing" not in c


def test_add_loss_keeps_the_graph():
    w = torch.tensor(2.0, requires_grad=True)
    c = LossContainer()
    c.add_loss("main", w * 3)
    c.add_loss("main", w)
    c.get_loss().backward()
    assert w.grad.item() == pytest.approx(4.0)


def test_add_loss_container_merges_by_key():
    a = LossContainer(router=1.0)
    b = LossContainer(router=0.5, controller=2.0)
    b.add_loss("main", 3.0)

    a.add_loss_container(b)

    assert list(a.loss_dict) == ["main", "router", "controller"]
    assert a.get_loss("main").item() == pytest.approx(3.0)
    assert a.get_loss("router").item() == pytest.approx(1.5)
    assert a.get_loss("controller").item() == pytest.approx(2.0)
    # The source container is left as it was.
    assert b.get_loss("router").item() == pytest.approx(0.5)
