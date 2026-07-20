"""IEEE Std 80 grounding-grid reference formulas.

Source
------
IEEE Std 80, *IEEE Guide for Safety in AC Substation Grounding*. The
grid grounding-resistance estimate implemented here is **Sverak's
equation**, IEEE Std 80-2000 Eq. (57) — identical to IEEE Std 80-2013
Eq. (53) — which refines the earlier Laurent–Niemann expression by an
explicit burial-depth term.

Conventions
-----------
- SI units throughout: resistivity in Ω·m, lengths in m, areas in m².
- ``depth`` is the burial depth $h$ of the horizontal grid below the
  soil surface; ground rods (if any) are folded into
  ``total_conductor_length``.

Range of validity
-----------------
Sverak's formula assumes **uniform soil** and a horizontal grid. IEEE
Std 80 states it for burial depths $0.25\\,\\mathrm{m} \\le h \\le
2.5\\,\\mathrm{m}$ and quotes an accuracy of a few percent for typical
substation grids; the ``groundfield`` image backend reproduces it to
well under 1 % (``tests/test_ieee80_benchmark.py``).

Notes
-----
Pure-Python scalars, intentionally not vectorised — comparison tests
need clarity, not performance (mirrors
:mod:`groundfield.references.dwight1936`).
"""

from __future__ import annotations

import math

__all__ = ["grid_resistance_sverak"]


def grid_resistance_sverak(
    rho: float,
    total_conductor_length: float,
    area: float,
    depth: float,
) -> float:
    r"""Grid grounding resistance after Sverak (IEEE Std 80 Eq. 57).

    $$
    R_g = \rho\left[\frac{1}{L_T}
          + \frac{1}{\sqrt{20\,A}}
            \left(1 + \frac{1}{1 + h\sqrt{20/A}}\right)\right]
    $$

    Parameters
    ----------
    rho
        Uniform soil resistivity $\rho$ in Ω·m.
    total_conductor_length
        Total buried length $L_T$ of all grid conductors (plus any
        ground rods) in metres.
    area
        Area $A$ of ground occupied by the grid in m².
    depth
        Burial depth $h$ of the grid in metres. IEEE Std 80 states the
        formula for $0.25 \le h \le 2.5$ m.

    Returns
    -------
    float
        Grid grounding resistance $R_g$ in Ω.

    Notes
    -----
    In the two limits the formula behaves as expected: $h \to \infty$
    removes the second bracket term (deep grid, no surface image boost)
    while $h \to 0$ doubles it (grid at the surface). The $1/L_T$ term
    is the conductor's own spreading contribution and vanishes for a
    very dense grid.
    """
    if rho <= 0.0 or total_conductor_length <= 0.0 or area <= 0.0:
        raise ValueError("rho, total_conductor_length and area must be positive.")
    if depth <= 0.0:
        raise ValueError("depth must be positive.")
    L_T, A, h = total_conductor_length, area, depth
    return rho * (
        1.0 / L_T
        + 1.0 / math.sqrt(20.0 * A) * (1.0 + 1.0 / (1.0 + h * math.sqrt(20.0 / A)))
    )
