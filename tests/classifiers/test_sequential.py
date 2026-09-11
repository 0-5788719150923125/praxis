"""praxis/classifiers/sequential.py: SequentialClassifier composition and its dashboards."""

import torch

from praxis import registry
from tests.stubs import Cfg, Enc


def _crystal_harmonic():
    torch.manual_seed(0)
    return registry.lookup("classifiers", "crystal_harmonic")(Cfg(), encoder=Enc())


def test_sequential_logits_match_manual_compose():
    """forward == terminal(transform(h)): the field is genuinely in the path."""
    classifier = _crystal_harmonic().eval()
    harmonic, crystal = classifier.stages[0], classifier.stages[1]
    feat = torch.randn(2, 6, Cfg.hidden_size)
    with torch.no_grad():
        composed = classifier(feat)
        manual = crystal(harmonic.transform(feat))
    assert torch.allclose(composed, manual, atol=1e-5)
    assert composed.shape == (2, 6, Cfg.vocab_size)


def test_crystal_harmonic_descriptions_are_unprefixed():
    """A single-field stack surfaces its stages' bare keys - the ``p{i}_``
    namespacing belongs to ParallelClassifier arms only."""
    descs = _crystal_harmonic().all_metric_descriptions()
    assert "harmonic_amplitudes_norm" in descs
    assert not any(k.startswith("p0_") for k in descs)
