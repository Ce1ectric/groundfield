"""Analytical reference formulas for solver validation.

This subpackage collects closed-form expressions from the classical
literature on grounding-resistance computation in **homogeneous** soil.
They serve two purposes:

1. **Plausibility checks for the numerical backends** — every new
   electrode geometry should ship with at least one comparison test
   against a closed-form formula.
2. **Sanity checks on 2-layer models that are nearly homogeneous**
   ($\\rho_1 \\approx \\rho_2$): the numerical result must converge to
   the homogeneous limit of the corresponding reference formula.

Modules
-------
dwight1936
    Formulas from Dwight, H. B.: *Calculation of Resistances to
    Ground*, Electrical Engineering / AIEE Transactions,
    December 1936, pp. 1319-1328. Table I covers driven rod, rod
    pair (close / far), buried horizontal wire, right-angle, 3/4/6/8
    point stars, ring, strip, round and vertical plate.
oeding
    Per-unit-length loop self- and mutual-inductance formulas from
    Oeding & Oswald (2016) *Elektrische Kraftwerke und Netze*
    (Springer), chapter 9 — used as an analytical reference for
    the segment-based Neumann implementation in
    :mod:`groundfield.coupling.inductance`.
carson
    Worked examples and tabulated values from Carson 1926
    (Bell STJ 5(4)) plus modern textbook reference points
    (Tleis 2008). Used to validate the Carson earth-return
    correction in :mod:`groundfield.coupling.carson` (ADR-0005).
earth_return
    Carson's engineering-level (Ω/km) earth-return **line impedance**
    — self and mutual series impedance with the equivalent-depth
    :math:`D_e` reactance and the soil-independent
    :math:`\\omega\\mu_0/8` resistance. Benchmarks the assembled
    Sommerfeld/Pollaczek inductive stack
    (``earth_inductive_model="sommerfeld"``).
ieee80
    Sverak's grid grounding-resistance formula from IEEE Std 80
    (*IEEE Guide for Safety in AC Substation Grounding*, 2000
    Eq. 57 / 2013 Eq. 53). Benchmarks the image backend's grid
    resistance against the industry-standard engineering estimate.
"""

from __future__ import annotations

from groundfield.references import (
    carson,
    dwight1936,
    earth_return,
    ieee80,
    oeding,
)

__all__ = ["carson", "dwight1936", "earth_return", "ieee80", "oeding"]
