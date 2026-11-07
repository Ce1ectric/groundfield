"""Touch and step voltages for grounding-system safety assessment.

This module turns a :class:`groundfield.solver.result.FieldResult` into
the engineering quantities used in the grounding-system safety
verification according to EN 50522:2022 (DIN EN 50522 (VDE 0101-2):
2023-10) / IEC 61936-1: the **touch voltage** :math:`U_T` and the
**step voltage** :math:`U_S`. It also provides the **permissible**
values :math:`U_{Tp}(t_f)` of EN 50522:2022, Table B.4 (the curve of
Figure 8), the permissible **prospective** (open-circuit) touch voltage
:math:`U_{vTp}` of Annex B.4, and the true-RMS touch voltage over a
fault-current spectrum.

Mathematical / physical content
-------------------------------
A grounding cluster at earth potential rise

.. math::

    U_E \\;=\\; \\varphi_\\text{cluster},

is connected — through any galvanically bonded metallic part —
to whatever a person can touch. Standing on the soil surface 1 m away
from the touched part (the conventional position of EN 50522), the
person's feet sit at the surface potential
:math:`\\varphi(\\mathbf{r}_\\text{feet})`. The prospective touch
voltage is

.. math::

    U_T \\;=\\; U_E \\;-\\; \\varphi(\\mathbf{r}_\\text{feet}).

Since review pass 10 (10.21) the metre is measured from the electrode's
**footprint** — the horizontal projection of its wires — not from its
connection point: the feet stand ``distance`` metres from the nearest
wire, on the side given by ``direction`` (seen from the length-weighted
centroid of the wires). Inside a closed loop the surface potential
sags, so :func:`touch_voltage_envelope` also evaluates the centre of
every closed mesh in open ground — the mesh centres of grids and the
centre of a ring, the analogue of the IEEE Std 80 *mesh voltage* — but
not, by default, the inside of a closed polyline, the building-outline
foundation of the generators (``interior=``).

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
* Frequency: quasi-static envelope :math:`f \\le 1\\,\\mathrm{kHz}`,
  inherited from the underlying Green's function.
* Current ansatz: the ``image`` family (``image``, ``image_2layer``,
  ``image_nlayer``, ``cim``) solves **one uniform current per
  electrode**. Impedances are fine, but near grids, meshes, stars, strip
  ends and bonded multi-electrode clusters the surface potential is
  impedance-grade, not field-grade. Measured against the per-segment
  ``mom`` backend (homogeneous 100 Ω·m, review pass 10): 20 m × 20 m
  grid with 4 × 4 meshes, 1 m outside the corner, touch +19 %, step
  −21 % (non-conservative); 4-arm star 1 m beyond a tip touch +7 %,
  step −10 %; 10 m strip 1 m beyond its end step −7 %; rings, rods and
  closed rectangular loops agree within 1 %. The helpers emit
  :class:`UniformCurrentFieldWarning` once per call when a uniform-current
  result is evaluated within :data:`NEAR_FIELD_DISTANCE_M` of such an
  electrode; re-solve with ``backend="mom"`` for field-grade values.
* Coordinate convention: the soil surface is at :math:`z = 0`,
  positive :math:`z` points downwards into the soil (see
  :mod:`groundfield.geometry.electrodes`). The default
  ``surface_z = 0.0`` therefore models bare-foot contact at the
  ground surface.
* :func:`permissible_touch_voltage_en50522` interpolates
  EN 50522:2022, Table B.4 for fault durations
  :math:`50\\,\\mathrm{ms} \\le t_f \\le 10\\,\\mathrm{s}` and clamps
  outside; the standard allows 80 V for durations considerably longer
  than 10 s (``long_duration=True``).

References
----------
- EN 50522:2022 — Earthing of power installations exceeding 1 kV a.c.,
  Annex B (Table B.1 body current, Table B.4 permissible touch voltage,
  B.4 prospective touch voltage). German edition DIN EN 50522
  (VDE 0101-2):2023-10.
- IEC 60479-1:2018 — Effects of current on human beings and livestock
  (curve c2, body impedance).
- IEC 61936-1:2021 — Power installations exceeding 1 kV a.c.
- IEEE Std 80-2013 — Guide for safety in AC substation grounding.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from groundfield.postprocess._footprint import footprint, interior_points

if TYPE_CHECKING:  # pragma: no cover
    from groundfield.solver.result import FieldResult
    from groundfield.world import World

__all__ = [
    "touch_voltage",
    "step_voltage",
    "touch_voltage_envelope",
    "touch_voltage_spectrum",
    "permissible_touch_voltage_en50522",
    "permissible_prospective_touch_voltage_en50522",
    "UniformCurrentFieldWarning",
    "NEAR_FIELD_DISTANCE_M",
]

Anchor = Literal["footprint", "connection_point"]


class UniformCurrentFieldWarning(UserWarning):
    """Field quantity from a uniform-current backend near a crowding geometry.

    The ``image`` family assigns one uniform current density to every
    electrode. Near grids, meshes, stars, strip ends and bonded
    multi-electrode clusters the true current crowds to edges, corners and
    tips, so touch and step voltages computed there are impedance-grade
    (review pass 10, 10.22; see the module docstring for measured
    deviations). Re-solve with ``backend="mom"`` for field-grade values,
    or silence the category with
    ``warnings.simplefilter("ignore", UniformCurrentFieldWarning)``.
    """


#: Backends that solve one uniform current per electrode.
_UNIFORM_CURRENT_BACKENDS = frozenset({"image", "image_2layer", "image_nlayer", "cim"})

#: Horizontal distance in m from a crowding electrode within which the
#: safety helpers warn about uniform-current results (10.22). At 5 m from
#: the footprint the measured touch / step deviations of the uniform
#: ansatz against ``mom`` are below 5 % for stars, strips, bonded
#: clusters and the default house grid; only large grids keep up to 8 %
#: (20 m grid corner, step voltage).
NEAR_FIELD_DISTANCE_M: float = 5.0


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _normalised(v: tuple[float, float, float]) -> np.ndarray:
    """Return ``v`` normalised to unit length, raise on the zero vector."""
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 0.0:
        raise ValueError(f"direction must be a non-zero 3-vector, got {tuple(v)}.")
    return arr / n


def _horizontal(direction: tuple[float, float, float]) -> np.ndarray:
    """Unit horizontal direction ``(ex, ey, 0)`` of ``direction``."""
    e_hat = _normalised(direction)
    e_h = np.array([e_hat[0], e_hat[1], 0.0])
    if float(np.linalg.norm(e_h)) <= 0.0:
        raise ValueError(f"direction has no horizontal component: {tuple(direction)}.")
    return e_h / float(np.linalg.norm(e_h))


def _check_distance(distance: float) -> None:
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"distance must be > 0, got {distance!r}.")


def _check_anchor(anchor: str) -> None:
    if anchor not in ("footprint", "connection_point"):
        raise ValueError(
            f"anchor must be 'footprint' or 'connection_point', got {anchor!r}."
        )


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


def _feet_xy(
    world: "World",
    electrode: str,
    directions: np.ndarray,
    distance: float,
    anchor: str,
) -> np.ndarray:
    """Horizontal foot positions ``(n, 2)`` for unit ``directions`` ``(n, 2)``."""
    e = world.get_electrode(electrode)
    if anchor == "connection_point":
        cp = e.connection_point
        return np.column_stack(
            [cp[0] + distance * directions[:, 0], cp[1] + distance * directions[:, 1]]
        )
    return footprint(e).feet(directions, distance)


def _crowding(electrode, result: "FieldResult") -> bool:
    """Whether the uniform-current ansatz misrepresents the field near it."""
    kind = getattr(electrode, "kind", None)
    if kind in ("grid_mesh", "mesh", "star", "strip"):
        return True
    if kind == "polyline" and not (electrode.closed and len(electrode.vertices) > 2):
        return True
    return len(result.clusters.get(electrode.name, [electrode.name])) > 1


def _uniform_current_caveat(
    result: "FieldResult",
    world: "World | None",
    points: np.ndarray,
    *,
    caller: str,
) -> None:
    """Warn once if a uniform-current result is evaluated near crowding wires."""
    if result.backend not in _UNIFORM_CURRENT_BACKENDS:
        return
    xy = np.asarray(points, dtype=float)[:, :2]
    near = False
    if world is None:
        # Without the world the electrode shapes are unknown: use the
        # point sources as the wire positions (conservative).
        if result.point_sources:
            src = np.array([ps.position[:2] for ps in result.point_sources])
            half = np.array([0.5 * ps.length for ps in result.point_sources])
            d = np.sqrt(((xy[:, None, :] - src[None, :, :]) ** 2).sum(-1)) - half
            near = bool(d.min() <= NEAR_FIELD_DISTANCE_M)
    else:
        for e in world.electrodes:
            if e.name not in result.electrode_potentials or not _crowding(e, result):
                continue
            try:
                fp = footprint(e)
            except TypeError:  # unknown kind: assume crowding everywhere
                near = True
                break
            if float(fp.distance(xy).min()) <= NEAR_FIELD_DISTANCE_M:
                near = True
                break
    if near:
        warnings.warn(
            f"{caller}: uniform-current backend '{result.backend}' within "
            f"{NEAR_FIELD_DISTANCE_M:g} m of a grid, mesh, star, strip, open "
            "polyline or bonded multi-electrode cluster: the surface potential "
            "is impedance-grade (20 m grid, 1 m outside the corner: touch "
            "+19 %, step -21 % against 'mom'). Use backend='mom' for "
            "field-grade touch and step voltages.",
            UniformCurrentFieldWarning,
            stacklevel=3,
        )


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
    anchor: Anchor = "footprint",
) -> complex:
    """Touch voltage at a single point on the soil surface.

    Evaluates :math:`U_T = U_E - \\varphi(\\mathbf{r}_\\text{feet})`,
    where :math:`U_E` is the cluster potential of the touched electrode.
    With the default ``anchor="footprint"`` the feet stand ``distance``
    metres (horizontally) from the nearest wire of the electrode, on the
    ray that leaves the electrode's footprint in the horizontal
    projection of ``direction``: 1 m outside the edge of a grid, 1 m
    beyond the tip of a star arm, 1 m outside a ring on either side.

    Parameters
    ----------
    result
        Solver output from :meth:`Engine.solve`.
    world
        Companion world used to resolve the electrode geometry.
    electrode
        Name of the touched electrode. Its cluster potential is the
        EPR seen by the hand.
    distance
        Horizontal distance between the touched part and the feet,
        in m. EN 50522 uses ``1.0``.
    direction
        Side of the electrode on which the person stands, in world
        coordinates, seen from the length-weighted centroid of the wires.
        Only the horizontal projection is used (the feet stay on the
        surface). Defaults to ``+x``.
    surface_z
        :math:`z` coordinate of the soil surface in m. Default ``0.0``
        (groundfield convention; positive ``z`` is below ground).
    frequency_index
        Index into :attr:`FieldResult.frequencies`. Default 0.
    anchor
        ``"footprint"`` (default since review pass 10, 10.21) measures
        ``distance`` from the nearest wire; ``"connection_point"``
        reproduces v0.15.0 and measures it from
        :attr:`connection_point` (the centre of a grid or star).

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
        If ``distance`` is not strictly positive, ``direction`` has no
        horizontal component, or ``anchor`` is unknown.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, for a uniform-current result evaluated near a
        crowding geometry (see the module docstring).

    Notes
    -----
    Up to v0.15.0 the metre was measured from the connection point, so
    for a 20 m × 20 m grid the feet stood above the grid and the helper
    reported 9.92 V where 1 m outside the edge has 69.6 V (100 A,
    100 Ω·m; F13 / 10.21).
    """
    _check_distance(distance)
    _check_anchor(anchor)
    e_h = _horizontal(direction)
    feet_xy = _feet_xy(world, electrode, e_h[None, :2], distance, anchor)[0]
    feet = np.array([[feet_xy[0], feet_xy[1], surface_z]])
    _uniform_current_caveat(result, world, feet, caller="touch_voltage")
    phi_feet = complex(result.potential(feet, frequency_index=frequency_index)[0])
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
    anchor: Anchor = "footprint",
    interior: bool | None = None,
    return_points: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Touch voltages around an electrode and inside its closed loops.

    Walks ``n_angles`` equally spaced directions and places the feet as
    in :func:`touch_voltage`: with ``anchor="footprint"`` this traces
    the curve at ``distance`` from the electrode's wires (rounded around
    corners and arm tips). Inside a closed loop the surface potential
    sags, so the touch voltage there can exceed every value on that curve
    (a 20 m × 20 m perimeter loop at 0.5 m depth, 100 Ω·m, 100 A: 158 V
    in the centre against at most 141 V 1 m outside). The envelope therefore also evaluates the
    centre of every closed mesh, the analogue of the *mesh voltage* of
    IEEE Std 80 (the worst touch voltage inside a mesh of a grid, near
    the mesh centre); see ``interior``. The maximum of ``|U_T|`` is the
    conservative envelope used in safety verification.

    Parameters
    ----------
    result, world
        See :func:`touch_voltage`.
    electrode
        Name of the touched electrode.
    distance
        Horizontal distance from the wires in m (1.0 by EN 50522).
    n_angles
        Number of equally spaced directions
        :math:`\\theta_k = 2\\pi k / n` with :math:`k = 0, \\dots, n - 1`.
        Must be ``>= 3``.
    surface_z, frequency_index, anchor
        See :func:`touch_voltage`.
    interior
        Which closed loops are sampled inside (review pass 10, 10.21).
        ``None`` (default): loops in open ground — the centre of every
        mesh of a ``mesh`` / ``grid_mesh`` electrode and the centre of a
        ``ring``; a closed ``polyline``, the building-outline foundation
        of the generators, is not sampled, because inside it lies the
        building. ``True``: also a closed polyline, at the point inside
        it farthest from its wires. ``False``: the offset curve only.
        The default cannot see buildings: pass ``interior=False`` where
        the inside of a ring or grid is not accessible ground (an
        axis-aligned generated foundation is a ``grid_mesh``; a ring may
        surround a station housing).
    return_points
        Also return the foot positions.

    Returns
    -------
    angles : np.ndarray, shape (n,)
        Azimuth in radians of each foot position, measured from ``+x``.
        The first ``n_angles`` entries are :math:`\\theta_k`; appended
        interior points carry their azimuth seen from the centroid of
        the wires (0 for a point that coincides with it).
    voltages : np.ndarray, shape (n,), dtype complex
        Touch-voltage phasor :math:`U_T` in V at every foot position.
    points : np.ndarray, shape (n, 3)
        Foot positions in m; only with ``return_points=True``.

    Raises
    ------
    ValueError
        For ``n_angles < 3``, a non-positive ``distance``, an unknown
        ``anchor`` or an ``interior`` other than ``None``, ``True`` and
        ``False``.

    Notes
    -----
    ``n = n_angles`` plus the number of interior points. To reproduce
    v0.15.0 (circle around the connection point) pass
    ``anchor="connection_point", interior=False``.
    """
    if n_angles < 3:
        raise ValueError(f"n_angles must be >= 3, got {n_angles}.")
    _check_distance(distance)
    _check_anchor(anchor)
    if interior not in (None, True, False):
        raise ValueError(f"interior must be None, True or False, got {interior!r}.")

    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    dirs = np.column_stack([np.cos(angles), np.sin(angles)])
    feet_xy = _feet_xy(world, electrode, dirs, distance, anchor)
    e = world.get_electrode(electrode)
    if interior is not False:
        inner = interior_points(e, outlines=bool(interior))
        if len(inner):
            ref = footprint(e).reference_point
            az = np.mod(
                np.arctan2(inner[:, 1] - ref[1], inner[:, 0] - ref[0]), 2 * np.pi
            )
            az[np.hypot(*(inner - ref).T) <= 1e-12] = 0.0
            angles = np.concatenate([angles, az])
            feet_xy = np.vstack([feet_xy, inner])
    feet = np.column_stack([feet_xy, np.full(len(feet_xy), surface_z, dtype=float)])
    _uniform_current_caveat(result, world, feet, caller="touch_voltage_envelope")
    phi_feet = result.potential(feet, frequency_index=frequency_index)
    U_E = _cluster_potential(result, electrode, frequency_index)
    voltages = U_E - phi_feet
    if return_points:
        return angles, voltages, feet
    return angles, voltages


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
    world: "World | None" = None,
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
        First foot position :math:`(x, y, z)` in m. The :math:`z`
        coordinate is overridden by ``surface_z`` if the latter is
        given; the default keeps ``z`` as supplied so callers can use
        the helper for buried-step studies if desired.
    direction
        Step direction in world coordinates. The horizontal
        projection is used (the :math:`z` component is ignored —
        both feet stay on the same surface).
    step
        Step length :math:`d_\\text{step}` in m. EN 50522 uses ``1.0``.
    surface_z
        Optional explicit surface :math:`z` in m. ``None`` keeps the
        :math:`z` of ``position`` (typical for bare-foot contact
        at the ground surface, ``position = (x, y, 0.0)``).
    frequency_index
        Index into :attr:`FieldResult.frequencies`.
    world
        Optional companion world. Only used to decide whether to emit
        :class:`UniformCurrentFieldWarning`: with the world the warning
        is restricted to the electrode shapes with measured ansatz
        errors; without it any wire within
        :data:`NEAR_FIELD_DISTANCE_M` of a uniform-current result
        triggers it.

    Returns
    -------
    complex
        Phasor :math:`U_S(f)` in V.

    Raises
    ------
    ValueError
        If ``step`` is not strictly positive or ``direction`` has
        no horizontal component.

    Warns
    -----
    UniformCurrentFieldWarning
        Once per call, see the module docstring. At the corner of a
        20 m × 20 m grid the uniform-current ansatz under-estimates the
        step voltage by 21 % (1 m outside) to 28 % (above the corner).
    """
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(f"step must be > 0, got {step!r}.")

    e_h = _horizontal(direction)
    z0 = float(position[2]) if surface_z is None else float(surface_z)
    p1 = np.array([float(position[0]), float(position[1]), z0])
    p2 = p1 + step * e_h
    pts = np.stack([p1, p2])
    _uniform_current_caveat(result, world, pts, caller="step_voltage")

    phi = result.potential(pts, frequency_index=frequency_index)
    return complex(phi[0] - phi[1])


# ---------------------------------------------------------------------
# Touch voltage over a fault spectrum
# ---------------------------------------------------------------------


def _source_current(result: "FieldResult", source: str | None, k: int) -> complex:
    """Impressed current of the selected current source at index ``k``."""
    if not result.sources:
        raise ValueError(
            "touch_voltage_spectrum needs the source snapshot of the solve "
            "(FieldResult.sources); this result carries none."
        )
    current = [s for s in result.sources if s.kind == "current"]
    if source is None:
        if len(current) != 1:
            names = ", ".join(s.name for s in current) or "none"
            raise ValueError(
                f"touch_voltage_spectrum: {len(current)} current sources "
                f"({names}); pass source=<name>."
            )
        record = current[0]
    else:
        matches = [s for s in current if s.name == source]
        if not matches:
            raise KeyError(f"No current source {source!r} in the result.")
        record = matches[0]
    i_src = complex(record.current[k])
    if i_src == 0:
        raise ValueError(
            f"Source {record.name!r} carries no current at frequency index {k}."
        )
    return i_src


def touch_voltage_spectrum(
    result: "FieldResult | Sequence[FieldResult]",
    world: "World",
    *,
    electrode: str,
    currents: "Sequence[complex] | Mapping[float, complex]",
    distance: float = 1.0,
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    surface_z: float = 0.0,
    anchor: Anchor = "footprint",
    source: str | None = None,
) -> float:
    """True-RMS touch voltage of a fault-current spectrum at one position.

    The field problem is linear, so the touch voltage of harmonic
    :math:`k` scales with the fault current :math:`I_k` of that
    harmonic. With :math:`U_T(f_k)` from a solve with source current
    :math:`I_\\text{src}(f_k)`, the touch voltage at the fixed foot
    position of :func:`touch_voltage` is

    .. math::

        U_{T,\\text{rms}} \\;=\\; \\sqrt{\\sum_k
        \\left|U_T(f_k)\\,\\frac{I_k}{I_\\text{src}(f_k)}\\right|^2},

    the RMS value of the superposed harmonics (they are orthogonal over
    a period).

    Parameters
    ----------
    result
        One multi-frequency result, or a sequence of results whose
        frequencies are taken in order (for example one result per
        harmonic).
    world
        Companion world (geometry of the touched electrode).
    electrode
        Name of the touched electrode.
    currents
        Fault current per frequency as RMS phasors (or RMS magnitudes)
        in A: either a sequence aligned with the frequencies of
        ``result``, or a mapping ``{frequency_Hz: current}`` that covers
        every frequency exactly once.
    distance, direction, surface_z, anchor
        Foot position, see :func:`touch_voltage`; the same position is
        used for every frequency.
    source
        Name of the current source the results were solved with; may be
        omitted when there is exactly one.

    Returns
    -------
    float
        RMS touch voltage :math:`U_{T,\\text{rms}}` in V.

    Raises
    ------
    ValueError
        If the currents do not match the frequencies, are not finite, a
        result carries no source snapshot, or the source current is zero
        at some frequency.

    Notes
    -----
    EN 50522's permissible touch voltage is a 50 Hz curve; comparing a
    spectrum RMS with :math:`U_{Tp}` ignores the frequency factors of
    IEC 60479-2 (the body tolerates more current at higher frequencies),
    which is conservative.
    """
    results = [result] if hasattr(result, "frequencies") else list(result)
    pairs = [
        (res, k, float(f)) for res in results for k, f in enumerate(res.frequencies)
    ]
    freqs = [f for _, _, f in pairs]
    if isinstance(currents, Mapping):
        keys = [float(f) for f in currents]
        values = []
        used: set[int] = set()
        for f in freqs:
            hit = [
                i
                for i, key in enumerate(keys)
                if math.isclose(key, f, rel_tol=1e-9, abs_tol=1e-9)
            ]
            if len(hit) != 1:
                raise ValueError(
                    f"currents must give exactly one value for {f:g} Hz; "
                    f"keys: {sorted(keys)}."
                )
            used.add(hit[0])
            values.append(list(currents.values())[hit[0]])
        extra = [keys[i] for i in range(len(keys)) if i not in used]
        if extra:
            raise ValueError(f"currents has no result for frequencies {extra} Hz.")
    else:
        values = list(currents)
        if len(values) != len(pairs):
            raise ValueError(
                f"currents has {len(values)} entries but the results carry "
                f"{len(pairs)} frequencies."
            )
    i_k = np.asarray(values, dtype=complex)
    if not np.all(np.isfinite(i_k)):
        raise ValueError("currents must be finite.")

    _check_distance(distance)
    _check_anchor(anchor)
    e_h = _horizontal(direction)
    feet_xy = _feet_xy(world, electrode, e_h[None, :2], distance, anchor)[0]
    feet = np.array([[feet_xy[0], feet_xy[1], surface_z]])
    warned = set()
    total = 0.0
    for (res, k, _), i_fault in zip(pairs, i_k, strict=True):
        if res.backend not in warned:
            _uniform_current_caveat(res, world, feet, caller="touch_voltage_spectrum")
            warned.add(res.backend)
        u_t = _cluster_potential(res, electrode, k) - complex(
            res.potential(feet, frequency_index=k)[0]
        )
        total += abs(u_t * i_fault / _source_current(res, source, k)) ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------
# Permissible touch voltage U_Tp per EN 50522:2022, Table B.4
# ---------------------------------------------------------------------

# EN 50522:2022, Table B.4 (DIN EN 50522 (VDE 0101-2):2023-10): fault
# duration t_f in s -> permissible touch voltage U_Tp in V, the
# tabulated form of the curve in Figure 8 (values rounded to 5 V in the
# standard). Verified against the standard text in review pass 10
# (10.42). The plateau of 85 V at 5 s and 10 s is explicit.
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

# EN 50522:2022, Table B.1: fault duration t_f in s -> permissible body
# current I_B in A (curve c2 of IEC 60479-1:2018, probability of
# ventricular fibrillation below 5 %, left hand to both feet).
_EN50522_IB_GRID: tuple[tuple[float, float], ...] = (
    (0.05, 0.900),
    (0.10, 0.800),
    (0.20, 0.600),
    (0.50, 0.200),
    (1.00, 0.080),
    (2.00, 0.060),
    (5.00, 0.051),
    (10.00, 0.050),
)

#: Permissible touch voltage in V that EN 50522:2022 allows for current
#: flow durations considerably longer than 10 s.
_EN50522_TP_LONG_DURATION = 80.0


def _loglog(
    t: float, grid: tuple[tuple[float, float], ...], *, exact_anchors: bool = False
) -> float:
    """Piecewise log-log interpolation, clamped to the grid ends.

    ``exact_anchors`` returns the tabulated value itself at a tabulated
    abscissa (``exp(log(v))`` may differ from ``v`` by one ulp). It is off
    for :math:`U_{Tp}`, whose arithmetic is pinned by the golden values of
    v0.15.0.
    """
    ts = np.array([x for x, _ in grid], dtype=float)
    vs = np.array([v for _, v in grid], dtype=float)
    if t <= ts[0]:
        return float(vs[0])
    if t >= ts[-1]:
        return float(vs[-1])
    if exact_anchors and t in ts:
        return float(vs[int(np.flatnonzero(ts == t)[0])])
    return float(math.exp(np.interp(math.log(t), np.log(ts), np.log(vs))))


def permissible_touch_voltage_en50522(
    t_clear_s: float, *, long_duration: bool = False
) -> float:
    """Permissible touch voltage :math:`U_{Tp}(t_f)` per EN 50522:2022.

    Returns the maximum admissible touch voltage as a function of the
    fault duration :math:`t_f`, interpolated in EN 50522:2022, Table B.4
    (DIN EN 50522 (VDE 0101-2):2023-10), the tabulated form of the curve
    in Figure 8 of the standard:

    .. math::

        \\{(t_k, U_{Tp,k})\\} \\;=\\;
        \\{(0.05, 725),\\ (0.10, 655),\\ (0.20, 525),\\
        (0.50, 225),\\ (1.00, 115),\\ (2.00, 95),\\
        (5.00, 85),\\ (10.0, 85)\\},

    in s and V (rounded to 5 V in the standard). Between the anchors the
    helper interpolates linearly in :math:`\\log t`–:math:`\\log U`;
    outside the table it clamps to the end values.

    Parameters
    ----------
    t_clear_s
        Fault duration :math:`t_f` in s. Must be > 0; ``math.inf`` is
        accepted together with ``long_duration=True``.
    long_duration
        For current-flow durations considerably longer than 10 s the
        standard allows a permissible touch voltage of 80 V. ``True``
        returns that value and requires ``t_clear_s > 10``; the default
        keeps the 85 V end value of Table B.4 beyond 10 s.

    Returns
    -------
    float
        Permissible touch voltage :math:`U_{Tp}` in V.

    Raises
    ------
    ValueError
        If ``t_clear_s`` is not strictly positive, is infinite without
        ``long_duration=True``, or is at most 10 s with
        ``long_duration=True``.

    Notes
    -----
    Review pass 10 (10.42) checked the anchors against the standard
    text: they are the published values. A secondary reproduction quoted
    in the review (716 / 654 / 537 / 220 / 117 / 96 / 86 / 85 V) does not
    match EN 50522:2022. The curve is a design reference; compare it
    with :func:`touch_voltage` (or, with additional resistances, use
    :func:`permissible_prospective_touch_voltage_en50522`).
    """
    if long_duration:
        if not (t_clear_s > 10.0):
            raise ValueError(
                "long_duration=True applies to current-flow durations "
                f"considerably longer than 10 s, got t_clear_s={t_clear_s!r}."
            )
        return _EN50522_TP_LONG_DURATION
    if not math.isfinite(t_clear_s) or t_clear_s <= 0.0:
        hint = (
            " (pass long_duration=True for faults that are not cleared)"
            if t_clear_s == math.inf
            else ""
        )
        raise ValueError(f"t_clear_s must be > 0 and finite, got {t_clear_s!r}{hint}.")
    return _loglog(float(t_clear_s), _EN50522_TP_GRID)


def permissible_prospective_touch_voltage_en50522(
    t_f: float,
    *,
    R_H: float = 0.0,
    R_F1: float = 0.0,
    rho_s: float | None = None,
    HF: float = 1.0,
    long_duration: bool = False,
) -> float:
    """Permissible prospective touch voltage :math:`U_{vTp}` (EN 50522:2022, B.4).

    The prospective (open-circuit) touch voltage may exceed the
    permissible touch voltage across the body by the drop of the body
    current over the additional resistances (equivalent circuit of
    Figure B.2, Equation (A.3) of the standard):

    .. math::

        U_{vTp}(t_f) \\;=\\; U_{Tp}(t_f) \\;+\\;
        \\frac{I_B(t_f)}{HF}\\,R_A,
        \\qquad
        R_A = R_H + R_{F1} + R_{F2},
        \\qquad
        R_{F2} = 1.5\\,\\mathrm{m}^{-1}\\cdot\\rho_s,

    with :math:`U_{Tp}` from :func:`permissible_touch_voltage_en50522`
    (Table B.4) and the permissible body current :math:`I_B(t_f)` of
    Table B.1 (0.05 s: 900 mA, 0.1 s: 800 mA, 0.2 s: 600 mA,
    0.5 s: 200 mA, 1 s: 80 mA, 2 s: 60 mA, 5 s: 51 mA, 10 s: 50 mA;
    interpolated log-log and clamped like :math:`U_{Tp}`).

    Parameters
    ----------
    t_f
        Fault duration in s, > 0.
    R_H
        Hand resistance (e.g. gloves) in Ω, ``>= 0``.
    R_F1
        Resistance of the footwear in Ω, ``>= 0`` (EN 50522 quotes
        1000 Ω as an average for worn, damp shoes).
    rho_s
        Resistivity of the surface layer at the standing position in
        Ω·m; ``None`` (default) omits the standing-position resistance
        :math:`R_{F2}`.
    HF
        Heart-current factor of the current path, ``> 0``; 1 for left
        hand to feet, the path the standard uses here.
    long_duration
        Use the 80 V of :func:`permissible_touch_voltage_en50522` for
        durations considerably longer than 10 s (the body current is
        then held at its 10 s value, 50 mA).

    Returns
    -------
    float
        Permissible prospective touch voltage :math:`U_{vTp}` in V.

    Raises
    ------
    ValueError
        If a resistance or ``rho_s`` is negative or not finite, ``HF`` is
        not positive, or ``t_f`` is invalid (see
        :func:`permissible_touch_voltage_en50522`).

    Examples
    --------
    Figure B.3 of the standard, 0.5 s, no footwear, 500 Ω·m gravel
    (:math:`R_F = 750\\,\\Omega`): 225 V + 0.2 A · 750 Ω = 375 V.

    >>> permissible_prospective_touch_voltage_en50522(0.5, rho_s=500.0)
    375.0
    """
    for name, value in (("R_H", R_H), ("R_F1", R_F1)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0 Ohm, got {value!r}.")
    if rho_s is not None and (not math.isfinite(rho_s) or rho_s < 0.0):
        raise ValueError(f"rho_s must be finite and >= 0 Ohm m, got {rho_s!r}.")
    if not math.isfinite(HF) or HF <= 0.0:
        raise ValueError(f"HF must be finite and > 0, got {HF!r}.")
    u_tp = permissible_touch_voltage_en50522(t_f, long_duration=long_duration)
    i_b = _loglog(min(float(t_f), 10.0), _EN50522_IB_GRID, exact_anchors=True)
    r_a = R_H + R_F1 + (0.0 if rho_s is None else 1.5 * float(rho_s))
    return u_tp + i_b / HF * r_a
