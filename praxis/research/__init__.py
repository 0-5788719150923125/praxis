"""Research-paper framing: prose fragments gated by active components.

The paper in ``research/`` is a living document. Sections that describe a
specific component (the crystal head, the CALM codec) are only true when that
component is in the run being written up. This package declares those sections
as :class:`Fragment` objects gated by config, and ``tools/export_framing.py``
renders the ones whose components are active into ``research/framing.tex``.
"""

from praxis.research.framing import FRAMING, Fragment, active_fragments

__all__ = ["FRAMING", "Fragment", "active_fragments"]
