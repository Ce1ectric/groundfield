"""Inductance formulas from Oeding/Oswald 2016, *Elektrische Kraftwerke
und Netze*, chapter 9 (overhead lines) and chapter 16 (cables).

This module collects the **per-unit-length** loop self- and mutual
inductance formulas that the standard German power-engineering
reference uses to characterise long, parallel conductor systems.
They are the textbook results that the segment-based Neumann
implementation in :mod:`groundfield.coupling.inductance` must
reproduce in the limit $\\ell \\to \\infty$.

All formulas assume:

- straight, parallel conductors,
- conductor radius small compared to inter-conductor distance,
- non-magnetic conductors ($\\mu_r = 1$),
- DC / quasi-static current distribution
  (uniform across the cross section, internal-field contribution
  $\\mu_0 / (8\\pi)$ per wire, see ``include_internal`` knobs).

The earth-return path is *not* included here — those corrections
(Carson, Pollaczek) live in their own reference modules in a later
release.

References
----------
Oeding, D. & Oswald, B. R. (2016). *Elektrische Kraftwerke und
Netze*, 8th ed., Springer.

- §9.3, Eq. (9.8) / (9.9): mutual-inductance per unit length between
  two two-wire loops, with mean geometric distances.
- §9.3, Eq. (9.13b) / (9.13c): self-inductance per unit length of a
  two-wire loop.
- Anhang A.7: mean geometric distances for typical cross-sectional
  geometries (rod with itself: $g_{11} = r \\cdot e^{-1/4}$).
"""

from __future__ import annotations

import math

__all__ = [
    "MU_0",
    "loop_self_inductance_per_length",
    "loop_mutual_inductance_per_length",
    "two_wire_loop_radius",
    "internal_inductance_per_length",
]

# Vacuum permeability — kept local to avoid cross-module imports for
# this small reference helper.
MU_0 = 4.0e-7 * math.pi


def two_wire_loop_radius(wire_radius: float, *, include_internal: bool = True) -> float:
    """Effective wire radius $r' = r \\cdot e^{-1/4}$ used in the
    Oeding loop-inductance formulas.

    The factor $e^{-1/4}$ is the **mean geometric distance** of a
    circular cross section from itself (Anhang A.7, case 1) and
    folds the internal-field contribution
    $\\mu_0 / (8\\pi)$ into a purely geometric expression. With
    $r' = r \\cdot e^{-1/4}$ the loop self-inductance becomes
    $L'_S = (\\mu_0 / \\pi) \\ln(d/r')$, which equals
    $(\\mu_0 / \\pi)(\\ln(d/r) + 1/4)$ — Eq. (9.13c).

    Pass ``include_internal=False`` to suppress the substitution and
    obtain the pure external-field expression.
    """
    if wire_radius <= 0.0:
        raise ValueError("wire_radius must be positive")
    if include_internal:
        return wire_radius * math.exp(-0.25)
    return wire_radius


def loop_self_inductance_per_length(
    distance: float,
    wire_radius: float,
    *,
    include_internal: bool = True,
) -> float:
    """Self-inductance per unit length of a two-wire loop.

    Implements Oeding/Oswald 2016 Eq. (9.13c):

    $$
    L'_S \\;=\\; \\frac{\\mu_0}{\\pi}
    \\Bigl[\\ln\\!\\Bigl(\\frac{d}{r}\\Bigr) + \\frac{1}{4}\\Bigr],
    $$

    valid for two long, straight, parallel wires of radius $r$ at
    centre-to-centre distance $d$, with $d \\gg r$.

    Parameters
    ----------
    distance
        Centre-to-centre distance $d$ between the two wires (m).
    wire_radius
        Wire radius $r$ (m).
    include_internal
        ``True`` (default) — include the internal-field
        contribution $\\mu_0 / (4\\pi)$ per loop. ``False`` —
        return the external-field expression
        $L'_{S,\\text{ext}} = (\\mu_0 / \\pi) \\ln(d/r)$ that
        applies in the high-frequency skin-effect limit.

    Returns
    -------
    L_per_length : float
        Loop self-inductance per unit length in H/m.
    """
    if distance <= 0.0 or wire_radius <= 0.0:
        raise ValueError("distance and wire_radius must be positive")
    if distance <= wire_radius:
        raise ValueError("distance must exceed wire_radius (d >> r assumed)")
    r_eff = two_wire_loop_radius(wire_radius, include_internal=include_internal)
    return (MU_0 / math.pi) * math.log(distance / r_eff)


def loop_mutual_inductance_per_length(
    *, s13: float, s14: float, s23: float, s24: float,
) -> float:
    """Mutual-inductance per unit length between two two-wire loops.

    Implements Oeding/Oswald 2016 Eq. (9.8):

    $$
    L'_{c,\\mathrm{I,II}} \\;=\\; \\frac{\\mu_0}{2\\pi}
    \\ln\\!\\Bigl(\\frac{s_{14}\\, s_{23}}{s_{13}\\, s_{24}}\\Bigr).
    $$

    Loop I consists of wires 1 and 2 (carrying currents $+I$ and
    $-I$); loop II consists of wires 3 and 4. The arguments are
    centre-to-centre distances between the indicated wire pairs;
    they must all be positive (no wire pair coincides — the
    self-inductance limit is handled by
    :func:`loop_self_inductance_per_length` instead).

    The same formula applies — with appropriate mean-geometric
    distances $g_{ij}$ in place of the Schwerpunktabstände
    $s_{ij}$ — to wires of finite cross section
    (Eq. 9.9 in Oeding/Oswald 2016). For circular cross sections
    with $r \\ll s$ the difference is negligible.

    Parameters
    ----------
    s13, s14, s23, s24
        Centre-to-centre distances in metres.

    Returns
    -------
    L_per_length : float
        Mutual inductance per unit length in H/m. Sign follows the
        wire-numbering convention (positive when both loops carry
        currents in the conventional ``+ — + —`` sense, i.e. wire
        1 is anti-parallel to wire 2 and wire 3 is anti-parallel to
        wire 4). The result can be negative when the loops are
        crossed.
    """
    for label, val in (("s13", s13), ("s14", s14), ("s23", s23), ("s24", s24)):
        if val <= 0.0:
            raise ValueError(f"{label} must be positive (got {val})")
    return (MU_0 / (2.0 * math.pi)) * math.log((s14 * s23) / (s13 * s24))


def internal_inductance_per_length() -> float:
    """Internal-field inductance per unit length of a single round
    conductor with uniform DC current distribution and
    $\\mu_r = 1$.

    Closed-form result $L'_\\text{int} = \\mu_0 / (8\\pi)$ — the
    classical textbook value, independent of the wire radius.

    Returns
    -------
    L_per_length : float
        $5.0 \\cdot 10^{-8}$ H/m, to numerical precision.
    """
    return MU_0 / (8.0 * math.pi)
