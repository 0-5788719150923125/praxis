"""Living-paper build: the generated inputs for ``research/main.tex``.

The paper is a living document whose data is regenerated from repository state.
Each concern is a module here, and ``python -m praxis.research.build`` runs them
all (the research-side analogue of ``praxis/web/src/build.py``):

- :mod:`runs` - recent validation curves -> variables.tex + data/run_*.csv
- :mod:`framing` - component-gated prose for the current run -> framing.tex
- :mod:`geometries` - Center PCA density figure -> geometries.tex + figures/
- :mod:`inlines` - single-value substitutions -> inlines.tex

Gated prose lives as :class:`Fragment` objects (config-keyed); inline edits as
:class:`~praxis.research.inlines.InlineEdit` objects (value providers).
"""

from praxis.research.framing import FRAMING, Fragment, active_fragments

__all__ = ["FRAMING", "Fragment", "active_fragments"]
