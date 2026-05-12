"""Touch and step voltages for grounding-system safety assessment.

This module turns a :class:`groundfield.solver.result.FieldResult` into
the engineering quantities used in the grounding-system safety
verification according to EN 50522:2010 / IEC 61936-1: the **touch
voltage** :math:`U_T` and the **step voltage** :math:`U_S`. It also
provides the corresponding **permissible** values
:math:`U_{TP}(t)` from EN 50522:2010, Figure B.3.

Mathematical / physical content
-------------------------------
A grounding cluster at earth potential rise

.. math::

    U_E \\;=\\; \\varphi_\\text{cluster},

is connected — through any galvanically bonded metallic part —
to whatever a person can touch. Standing on the soil surface
1 m away from the touched part (the conventional test point of
EN 50522), the person's feet sit at the surface potential
:math:`\\varphi(\\mathbf{r}_\\text{feet})`. The voltage
appearing across the body is

.. math::

    U_T \\;=\\; U_E \\;-\\; \\varphi(\\mathbf{r}_\\text{feet}).

For a *step* voltage the body bridges two surface points at the
typical step distance :math:`d_\\text{step} = 1\\,\\mathrm{m}`:

.. math::

    U_S \\;=\\; \\varphi(\\mathbf{r}_1) \\;-\\;
              \\varphi(\\mathbf{r}_1 + d_\\text{step}\\,\\hat{\\mathbf e}).

Both quantities are returned as complex phasors per frequency
index — consistent with the rest of ``groundfield`` — so that
inductive- and resistive-coupling effects above DC remain
visible.

Validity envelope
-----------------
* Frequency: dissertation envelope :math:`f \\le 1\\,\\mathrm{kHz}`,
  inherited from the underlying Green's function.
* Geometry: relies on
  :meth:`FieldResult.potential` and
  :meth:`FieldResult.electrode_potentials`. Works for every backend
  that populates ``point_sources`` (image, image_2layer,
  image_nlayer, mom, mom_sommerfeld, cim, bem) and is silently
  inapplicable to stub backends (raises through
  :meth:`FieldResult.potential`).
* Coordinate convention: the soil surface is at :math:`z = 0`,
  positive :math:`z` points downwards into the soil (see
  :mod:`groundfield.geometry.electrodes`). The default
  ``surface_z = 0.0`` therefore models bare-foot contact at the
  ground surface.
* Permissible-voltage helper
  :func:`permissible_touch_voltage_en50522` covers the
  low-voltage standard curve (EN 50522:2010, Fig. B.3) for
  fault-clearing times :math:`50\\,\\mathrm{ms} \\le t_F \\le 10\\,
  \\mathrm{s}`. Outside that range the value is clamped to the
  table endpoints, mirroring the stationary plateau the standard
  prescribes for long-duration faults.

References
----------
- EN 50522:2010 — Earthing of power installations exceeding
  1 kV a.c., Annex B (allowable touch voltages).
- IEC 61936-1:2014 — Power installations exceeding 1 kV a.c.
- IEEE Std 80-2013 — Guide for safety in AC substation grounding.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "touch_voltage",
    "step_voltage",
    "touch_voltage_envelope",
    "permissible_touch_voltage_en50522",
]


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _normalised(v: tuple[float, float, float]) -> np.ndarray:
    """Return ``v`` normalised to unit length, raise on the zero vector."""
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 0.0:
        raise ValueError(
            f"direction must be a non-zero 3-vector, got {tuple(v)}."
        )
    return arr / n


def _cluster_potential(
    result: "FieldResult", electrode: str, frequency_index: int
) -> complex:
    """Return the cluster potential the user is touching."""
    if electrode not in result.electrode_potentials:
        raise KeyError(
            f"Electrode '{electrode}' not found in result.electrode_potentials. "
            f"Known: {sorted(result.electrode_potentials.keys())}."
        )
    try:
        return complex(result.electrode_potentials[electrode][frequency_index])
    except IndexError as e:  # pragma: no cover — defensive
        raise IndexError(
            f"frequency_index {frequency_index} out of range "
            f"[0, {len(result.electrode_potentials[electrode])})."
        ) from e


# ---------------------------------------------------------------------
# Touch voltage
# ---------------------------------------------------------------------


def touch_voltage(
    result: "FieldResult",
    world: "World",
    *,
    electrode: str,
    distance: float = 1.0,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    surface_z: float = 0.0,
    frequency_index: int = 0,
) -> complex:
    """Touch voltage at a single point on the soil surface.

    Evaluates :math:`U_T = U_E - \\varphi(\\mathbf{r}_\\text{feet})`,
    where :math:`U_E` is the cluster potential of the touched
    electrode and :math:`\\mathbf{r}_\\text{feet}` is the surface
    point ``distance`` metres away from the electrode's
    :attr:`connection_point` in the (horizontal projection of the)
    given ``direction``.

    Parameters
    ----------
    result
        Solver output from :meth:`Engine.solve`.
    world
        Companion world used to resolve the electrode's connection
        point.
    electrode
        Name of the touched electrode. Its cluster potential is the
        EPR seen by the hand.
    distance
        Horizontal distance between the touched part and the feet,
        in metres. EN 50522 uses ``1.0``.
    direction
        Direction in which the feet sit, expressed in world
        coordinates. The horizontal projection of this vector is
        used (the :math:`z` component is ignored on purpose — feet
        stay on the surface). Defaults to ``+x``.
    surface_z
        :math:`z` coordinate of the soil surface in metres.
        Default ``0.0`` (groundfield convention; positive ``z`` is
        below ground).
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.

    Returns
    -------
    complex
        Phasor :math:`U_T(f) = U_E(f) - \\varphi(\\mathbf{r}_\\text{feet}, f)`
        in V.

    Raises
    ------
    KeyError
        If ``electrode`` is unknown to ``world`` or ``result``.
    ValueError
        If ``distance`` is not strictly positive or ``direction``
        is the zero vector after horizontal projection.
    """
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"distance must be > 0, got {distance!r}.")

    cp = world.get_electrode(electrode).connection_point
    e_hat = _normalised(direction)
    # Horizontal projection: enforce feet on the soil surface.
    e_h = np.array([e_hat[0], e_hat[1], 0.0])
    if float(np.linalg.norm(e_h)) <= 0.0:
        raise ValueError(
            f"direction has no horizontal component: {tuple(direction)}."
        )
    e_h = e_h / float(np.linalg.norm(e_h))

    feet = np.array([cp[0], cp[1], surface_z]) + distance * e_h
    phi_feet = complex(
        result.potential(feet[None, :], frequency_index=frequency_index)[0]
    )
    U_E = _cluster_potential(result, electrode, frequency_index)
    return U_E - phi_feet


def touch_voltage_envelope(
    result: "FieldResult",
    world: "World",
    *,
    electrode: str,
    distance: float = 1.0,
    n_angles: int = 24,
    surface_z: float = 0.0,
    frequency_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Touch-voltage profile around an electrode.

    Walks a horizontal circle of radius ``distance`` around the
    electrode's :attr:`connection_point` in ``n_angles`` equal
    angular steps and returns the touch voltage at every direction.
    The maximum of ``|U_T|`` is the conservative envelope used in
    safety verification.

    Parameters
    ----------
    result, world
        See :func:`touch_voltage`.
    electrode
        Name of the touched electrode.
    distance
        Radius of the circle in metres (1.0 by EN 50522).
    n_angles
        Number of equally spaced sample directions
        :math:`\\theta_k = 2\\pi k / n` with :math:`k = 0,
        \\dots, n - 1`. Must be ``>= 3``.
    surface_z, frequency_index
        See :func:`touch_voltage`.

    Returns
    -------
    angles : np.ndarray, shape (n_angles,)
        Angles :math:`\\theta_k` in radians, measured from the
        ``+x`` axis.
    voltages : np.ndarray, shape (n_angles,), dtype complex
        Touch-voltage phasor :math:`U_T(\\theta_k)` in V.
    """
    if n_angles < 3:
        raise ValueError(f"n_angles must be >= 3, got {n_angles}.")
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"distance must be > 0, got {distance!r}.")

    cp = world.get_electrode(electrode).connection_point
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    feet = np.column_stack(
        [
            cp[0] + distance * np.cos(angles),
            cp[1] + distance * np.sin(angles),
            np.full(n_angles, surface_z, dtype=float),
        ]
    )
    phi_feet = result.potential(feet, frequency_index=frequency_index)
    U_E = _cluster_potential(result, electrode, frequency_index)
    return angles, U_E - phi_feet


# ---------------------------------------------------------------------
# Step voltage
# ---------------------------------------------------------------------


def step_voltage(
    result: "FieldResult",
    *,
    position: tuple[float, float, float],
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    step: float = 1.0,
    surface_z: float | None = None,
    frequency_index: int = 0,
) -> complex:
    """Step voltage between two surface points.

    Evaluates :math:`U_S = \\varphi(\\mathbf{r}_1) -
    \\varphi(\\mathbf{r}_1 + d_\\text{step}\\,\\hat{\\mathbf{e}})`
    on the soil surface.

    Parameters
    ----------
    result
        Solver output.
    position
        First foot position :math:`(x, y, z)` in metres. The
        :math:`z` coordinate is overridden by ``surface_z`` if the
        latter is given; the default keeps ``z`` as supplied so
        callers can use the helper for buried-step studies if
        desired.
    direction
        Step direction in world coordinates. The horizontal
        projection is used (the :math:`z` component is ignored —
        both feet stay on the same surface).
    step
        Step length :math:`d_\\text{step}` in metres. EN 50522
        uses ``1.0``.
    surface_z
        Optional explicit surface :math:`z`. ``None`` keeps the
        :math:`z` of ``position`` (typical for bare-foot contact
        at the ground surface, ``position = (x, y, 0.0)``).
    frequency_index
        Index into :attr:`FieldResult.frequencies`.

    Returns
    -------
    complex
        Phasor :math:`U_S(f)` in V.

    Raises
    ------
    ValueError
        If ``step`` is not strictly positive or ``direction`` has
        no horizontal component.
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step must be > 0, got {step!r}.")

    e_hat = _normalised(direction)
    e_h = np.array([e_hat[0], e_hat[1], 0.0])
    if float(np.linalg.norm(e_h)) <= 0.0:
        raise ValueError(
            f"direction has no horizontal component: {tuple(direction)}."
        )
    e_h = e_h / float(np.linalg.norm(e_h))

    z0 = float(position[2]) if surface_z is None else float(surface_z)
    p1 = np.array([float(position[0]), float(position[1]), z0])
    p2 = p1 + step * e_h
    pts = np.stack([p1, p2])

    phi = result.potential(pts, frequency_index=frequency_index)
    return complex(phi[0] - phi[1])


# ---------------------------------------------------------------------
# Permissible touch voltage U_TP per EN 50522:2010, Table B.4
# ---------------------------------------------------------------------

# Anchor points of EN 50522:2010, **Table B.4** ("Berechnete Werte
# der zulässigen Berührungsspannung U_TP in Abhängigkeit von der
# Fehlerdauer t_F", values rounded to 5 V in the standard). This is
# the tabulated form of the same curve plotted in Figure B.3, and
# it is the form the standard treats as normative. The grid spans
# t_F in [50 ms, 10 s]; the standard's terminal plateau at 85 V is
# explicit (the values for 5 s and 10 s are identical).
_EN50522_TP_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 725.0),
    (0.10, 655.0),
    (0.20, 525.0),
    (0.50, 225.0),
    (1.00, 115.0),
    (2.00, 95.0),
    (5.00, 85.0),
    (10.00, 85.0),
)


def permissible_touch_voltage_en50522(t_clear_s: float) -> float:
    """Permissible touch voltage :math:`U_{TP}(t_F)` per EN 50522:2010.

    Returns the maximum admissible touch voltage as a function of
    the fault-clearing time :math:`t_F`. The reference values are
    taken **verbatim** from EN 50522:2010, Table B.4 — the
    normative tabulation of the curve in Figure B.3 (values
    rounded to 5 V in the standard).

    Mathematically the helper is a piecewise-loglog interpolation
    over the canonical anchor points

    .. math::

        \\{(t_k, U_{TP,k})\\} \\;=\\;
        \\{(0.05, 725),\\ (0.10, 655),\\ (0.20, 525),\\
        (0.50, 225),\\ (1.00, 115),\\ (2.00, 95),\\
        (5.00, 85),\\ (10.0, 85)\\},

    with units :math:`\\mathrm{s}` and :math:`\\mathrm{V}`. Outside
    the grid the values are clamped to the endpoints — the standard
    does not extend the curve below 50 ms, and the terminal plateau
    at 85 V is explicit (the table reports identical values at 5 s
    and 10 s).

    Parameters
    ----------
    t_clear_s
        Fault-clearing time :math:`t_F` in seconds. Must be > 0.

    Returns
    -------
    float
        Permissible touch voltage :math:`U_{TP}` in V.

    Raises
    ------
    ValueError
        If ``t_clear_s`` is not strictly positive.

    Notes
    -----
    The EN 50522 curve is intended for engineering design; for the
    AP1 dissertation the helper is used as a *reference line* on
    plots of computed touch voltages, not as a regulatory
    pass/fail. Use :func:`touch_voltage` to compute the actual
    :math:`U_T` and compare against this reference.
    """
    if not math.isfinite(t_clear_s) or t_clear_s <= 0.0:
        raise ValueError(f"t_clear_s must be > 0, got {t_clear_s!r}.")

    ts = np.array([t for t, _ in _EN50522_TP_GRID], dtype=float)
    us = np.array([u for _, u in _EN50522_TP_GRID], dtype=float)
    if t_clear_s <= ts[0]:
        return float(us[0])
    if t_clear_s >= ts[-1]:
        return float(us[-1])

    # Log-log linear interpolation.
    log_t = math.log(t_clear_s)
    log_ts = np.log(ts)
    log_us = np.log(us)
    return float(math.exp(np.interp(log_t, log_ts, log_us)))
