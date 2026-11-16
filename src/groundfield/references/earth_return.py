r"""Carson earth-return line-impedance reference (per unit length).

Closed-form series impedance of a conductor with **earth return**, as
derived by Carson (1926) and reproduced in every power-system textbook.
It is the engineering-level (Ω/km) companion to the dimensionless Carson
integral tabulated in :mod:`groundfield.references.carson`, and the
reference against which the assembled Sommerfeld/Pollaczek earth-return
stack (``earth_inductive_model="sommerfeld"``) is benchmarked in
``tests/test_earth_return_benchmark.py`` and
``notebooks/45_earth_return_line_impedance.ipynb``.

Mathematical background
-----------------------
For a thin conductor of geometric mean radius :math:`\mathrm{GMR}` above
or below a homogeneous earth of resistivity :math:`\rho`, carrying a
current that returns through the earth, Carson's series impedance per
unit length at angular frequency :math:`\omega = 2\pi f` is

.. math::

    Z'_\text{self} = \underbrace{\frac{\omega\mu_0}{8}}_{R'_\text{earth}}
        + j\,\frac{\omega\mu_0}{2\pi}\,\ln\!\frac{D_e}{\mathrm{GMR}},
    \qquad
    Z'_\text{mutual} = \frac{\omega\mu_0}{8}
        + j\,\frac{\omega\mu_0}{2\pi}\,\ln\!\frac{D_e}{d_{ij}},

where :math:`d_{ij}` is the conductor–conductor separation and

.. math::

    D_e = 658.87\,\sqrt{\rho / f}\ \text{[m]}

is Carson's **equivalent earth-return depth** — the depth of the
fictitious return conductor that reproduces the earth's inductive
effect. Two features carry the physics:

* the earth-return **resistance** :math:`R'_\text{earth} = \omega\mu_0/8`
  (:math:`\approx 0.0493` Ω/km at 50 Hz) is **independent of the soil
  resistivity** — it is the loss of the return current spreading through
  the earth;
* the equivalent depth :math:`D_e \propto \sqrt{\rho/f}` *shrinks* with
  frequency and *grows* with soil resistivity — a more resistive or
  lower-frequency earth lets the return current spread deeper and wider.
  The earth-return **reactance**
  :math:`X' = (\omega\mu_0/2\pi)\ln(D_e/\mathrm{GMR})` therefore
  *increases* with soil resistivity and also *increases* with frequency:
  the :math:`\omega` prefactor dominates the shrinking logarithm, so
  :math:`X'` rises almost linearly with :math:`f` (1000 Ω·m,
  GMR = 7 mm: 0.814 mΩ/m at 50 Hz → 14.39 mΩ/m at 1 kHz, a 17.7×
  rise over a 20× frequency step).

Internal-inductance convention (review pass 10, finding 10.46). The
formulas here are **external** to the conductor for a given
:math:`\mathrm{GMR}`. The assembled inductive stack of groundfield adds
the DC internal inductance :math:`\mu_0/(8\pi)` per metre of a solid round
conductor to every self term by default
(:func:`groundfield.coupling.inductance.thin_wire_self_inductance`,
``include_internal=True``), so a like-for-like comparison uses
:math:`\mathrm{GMR} = a\,e^{-1/4} = 0.7788\,a`; with ``GMR = a`` the
stack sits 2.0–2.5 % above Carson's reactance (100 Ω·m at 50 Hz to
30 Ω·m at 1 kHz, 4 mm wire) for that reason alone. Use ``GMR = a``
together with ``include_internal=False``. The frequency dependence of the
internal impedance (skin effect) is
:meth:`groundfield.conductors.Conductor.internal_impedance_per_length`.

References
----------
- Carson, J. R. (1926). Wave propagation in overhead wires with ground
  return. *Bell Syst. Tech. J.* **5**(4), 539–554.
- Tleis, N. D. (2008). *Power Systems Modelling and Fault Analysis*,
  Newnes, §3.2 (earth-return impedance, equivalent depth :math:`D_e`).
- Oeding, D. & Oswald, B. R. (2016). *Elektrische Kraftwerke und Netze*,
  8. Aufl., Springer, §9.4.
"""

from __future__ import annotations

import math

__all__ = [
    "MU_0",
    "carson_earth_return_resistance",
    "carson_equivalent_depth",
    "carson_self_impedance",
    "carson_mutual_impedance",
]

MU_0 = 4.0e-7 * math.pi
"""Vacuum permeability μ0 = 4π · 10⁻⁷ H/m (the conventional value)."""


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if not (value > 0.0) or not math.isfinite(value):
            raise ValueError(f"{name} must be positive and finite, got {value}")


def carson_earth_return_resistance(frequency: float) -> float:
    r"""Carson earth-return resistance per unit length :math:`\omega\mu_0/8`.

    Independent of the soil resistivity. Returns Ω/m.

    Parameters
    ----------
    frequency
        Frequency in Hz. Must be positive.

    Returns
    -------
    float
        Earth-return resistance in Ω/m (:math:`\approx 4.93\times10^{-5}`
        at 50 Hz, i.e. 0.0493 Ω/km).
    """
    _check_positive(frequency=frequency)
    return 2.0 * math.pi * frequency * MU_0 / 8.0


def carson_equivalent_depth(rho: float, frequency: float) -> float:
    r"""Carson equivalent earth-return depth :math:`D_e = 658.87\sqrt{\rho/f}`.

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m. Must be positive.
    frequency
        Frequency in Hz. Must be positive.

    Returns
    -------
    float
        Equivalent depth in metres.
    """
    _check_positive(rho=rho, frequency=frequency)
    return 658.87 * math.sqrt(rho / frequency)


def carson_self_impedance(rho: float, frequency: float, *, gmr: float) -> complex:
    r"""Carson earth-return **self** impedance per unit length (Ω/m).

    :math:`Z' = \omega\mu_0/8 + j\,(\omega\mu_0/2\pi)\ln(D_e/\mathrm{GMR})`,
    the external earth-return series impedance of a single conductor (no
    internal-conductor resistance or inductance).

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m. Must be positive.
    frequency
        Frequency in Hz. Must be positive.
    gmr
        Geometric mean radius of the conductor in m: the physical wire
        radius :math:`a` for an external-inductance model, ``0.7788·a``
        to include the internal inductance of a solid round conductor
        (the convention of the assembled stack, see the module
        docstring). Must be positive.

    Returns
    -------
    complex
        Series impedance :math:`R' + jX'` in Ω/m.
    """
    _check_positive(rho=rho, frequency=frequency, gmr=gmr)
    omega = 2.0 * math.pi * frequency
    r_earth = omega * MU_0 / 8.0
    d_e = carson_equivalent_depth(rho, frequency)
    x = omega * MU_0 / (2.0 * math.pi) * math.log(d_e / gmr)
    return complex(r_earth, x)


def carson_mutual_impedance(
    rho: float, frequency: float, *, separation: float
) -> complex:
    r"""Carson earth-return **mutual** impedance per unit length (Ω/m).

    :math:`Z' = \omega\mu_0/8 + j\,(\omega\mu_0/2\pi)\ln(D_e/d)`, the
    per-unit-length coupling between two parallel conductors a horizontal
    distance ``separation`` apart, both closing through the earth. This is
    the impedance behind an induced voltage on a parallel measurement /
    pilot line — Carson's original railway-induction problem.

    Parameters
    ----------
    rho
        Soil resistivity in Ω·m. Must be positive.
    frequency
        Frequency in Hz. Must be positive.
    separation
        Conductor–conductor distance in m. Must be positive.

    Returns
    -------
    complex
        Mutual impedance :math:`R' + jX'` in Ω/m.
    """
    _check_positive(rho=rho, frequency=frequency, separation=separation)
    omega = 2.0 * math.pi * frequency
    r_earth = omega * MU_0 / 8.0
    d_e = carson_equivalent_depth(rho, frequency)
    x = omega * MU_0 / (2.0 * math.pi) * math.log(d_e / separation)
    return complex(r_earth, x)
