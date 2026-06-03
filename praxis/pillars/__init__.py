"""Pillars: the components that justify Praxis's research from repository state.

Each pillar is a module that turns what the code is actually doing into a piece
of the living paper, regenerated on demand. They feed the standalone ``research/``
paper directory but are not coupled to it - this package knows Praxis, the paper
does not need to know this package. ``python -m praxis.pillars.build`` runs them
all (the research-side analogue of ``praxis/web/src/build.py``):

- :mod:`runs` - recent validation curves -> variables.tex + data/run_*.csv
- :mod:`framing` - component-gated prose for the current run -> framing.tex
- :mod:`geometries` - Center PCA density figure -> geometries.tex + figures/
- :mod:`halting` - halting-distribution figure -> halting.tex + figures/
- :mod:`inlines` - single-value substitutions -> inlines.tex
- :mod:`proofs` - formal claims attached to framings, consistency-checked

Gated prose lives as :class:`Fragment` objects (config-keyed); inline edits as
:class:`~praxis.pillars.inlines.InlineEdit` objects; proofs as
:class:`~praxis.pillars.proofs.Proof` objects linked back to the framings.
"""

from praxis.pillars.framing import FRAMING, Fragment, active_fragments

__all__ = ["FRAMING", "Fragment", "active_fragments"]
