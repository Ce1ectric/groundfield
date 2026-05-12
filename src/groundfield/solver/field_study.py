"""Top-level coordination class ``FieldStudy`` (placeholder).

A ``FieldStudy`` would bundle all inputs of a simulation run:

* the soil model (``groundfield.soil``),
* the grounding-system geometry (``groundfield.geometry``),
* optional conductors (``groundfield.conductors``),
* the frequency list,
* the boundary conditions (feed-in, auxiliary electrode, fault).

After ``solve()`` the results are available as a ``FieldResult`` and
can be further processed via ``groundfield.postprocess`` and
``groundfield.io``.

Notes
-----
The current preferred entry point is the lightweight pair ``World`` +
``Engine`` (see :mod:`groundfield.world` and
:mod:`groundfield.solver.engine`). The ``FieldStudy`` class is kept
as a placeholder for a possible future high-level wrapper that
combines both objects.
"""

from __future__ import annotations

__all__: list[str] = []
