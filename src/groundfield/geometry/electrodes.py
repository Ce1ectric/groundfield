"""Electrode geometries.

Defines the electrode primitives supported by the PDE / field model as
plain data containers (Pydantic v2). Each class describes the
geometric parameters (position, dimensions, wire radius). The
discretisation into segments is performed inside the solver backend.

Conventions
-----------
- Coordinate system: right-handed; the $z$ axis points
  **downwards** into the soil. The soil surface is at $z = 0$.
  A depth of 1 m therefore corresponds to $z = 1.0$ (positive
  depth values).
- Lengths in metres, wire radii in metres.

Notes
-----
For TN distribution networks, ``RodElectrode`` (driven rod),
``RingElectrode`` (foundation / ring earth electrode) and
``MeshElectrode`` (mesh / foundation earth electrode) are the most
relevant primitives.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from groundfield.geometry.primitives import (
    WireSegments,
    _edge_wires,
    _grid_wire_polylines,
    _grid_wires,
    _segment_count,
    _split_interval_at_interfaces,
    _straight_wire,
)

__all__ = [
    "Electrode",
    "RodElectrode",
    "RingElectrode",
    "StripElectrode",
    "PolylineElectrode",
    "StarElectrode",
    "MeshElectrode",
    "GridMeshElectrode",
    "concrete_shell_coefficient",
    "shell_coefficient_from_resistance",
    "shell_resistance_from_coefficient",
]

#: A finite float (NaN and infinities are rejected at construction).
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

#: A finite, strictly positive float (lengths and extents in m).
PositiveLength = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]

# 3-D coordinate (x, y, z) in m; z axis points into the soil.
Point3D = tuple[FiniteFloat, FiniteFloat, FiniteFloat]

#: Edges shorter than this (metres) are treated as degenerate: a
#: :class:`PolylineElectrode` rejects them so a duplicate vertex cannot
#: silently collapse a segment to zero length. Set far below any physical
#: wire radius, so only true coincidences trip it.
_MIN_EDGE_LENGTH_M = 1e-9

#: Interface slivers (review pass 10, 10.32): an interface cut that would
#: leave a part of a wire shorter than ``SLIVER_RADII`` wire radii (the wire
#: diameter) is dropped; the short part joins its neighbour, whose end
#: segment then reaches past the interface by less than one diameter and is
#: assigned to the layer of its midpoint. A wire piece shorter than its
#: diameter is outside the thin-wire model: up to 0.15.0 a rod whose tip lay
#: 1 mm below an interface raised a ``ValueError`` (segment shorter than the
#: wire radius) and one 9 mm below it acted as a short. The threshold does
#: not scale with the segment length on purpose: a 10 cm tip in a lower
#: layer 33 times more conductive carries a large share of the current, and
#: merging it at ``ds = 2 m`` (``ds/10``) put the resistance 122 % too high
#: (evidence ``e1032_sliver_threshold.py``).
SLIVER_RADII = 2.0


def _merge_interface_slivers(
    parts: list[tuple[float, float]], min_length: float
) -> list[tuple[float, float]]:
    """Drop interface cuts that leave a part shorter than ``min_length`` (m).

    Parameters
    ----------
    parts
        Consecutive intervals ``(t_lo, t_hi)`` in m along the wire, as
        returned by :func:`~groundfield.geometry.primitives._split_interval_at_interfaces`.
    min_length
        Shortest part in m that is kept.

    Returns
    -------
    list of tuple
        The intervals after dropping the cuts.
    """
    if len(parts) <= 1:
        return parts
    bounds = [parts[0][0]] + [hi for _, hi in parts]
    kept = [bounds[0]]
    for k in range(1, len(bounds) - 1):
        cut = bounds[k]
        if cut - kept[-1] < min_length or bounds[k + 1] - cut < min_length:
            continue
        kept.append(cut)
    kept.append(bounds[-1])
    return list(zip(kept[:-1], kept[1:], strict=True))


def _require_longer_than_radius(
    electrode: "_ElectrodeBase", dimension: float, what: str
) -> None:
    """Reject a dimension that is not longer than the wire radius.

    A thin-wire electrode must be much longer than it is thick; a
    dimension at or below the wire radius is degenerate (and most often a
    unit mix-up, e.g. a wire radius typed in mm). Detected at
    construction instead of as a thin-wire error of one segment at solve
    time (review pass 10, 10.67).

    Raises
    ------
    ValueError
        If ``dimension <= electrode.wire_radius``.
    """
    if not dimension > electrode.wire_radius:
        raise ValueError(
            f"{type(electrode).__name__} {electrode.name!r}: its {what} "
            f"({dimension:g} m) is not longer than its wire_radius "
            f"({electrode.wire_radius:g} m). All lengths are in metres; a wire "
            "radius of 5 mm is wire_radius=0.005."
        )


# ---------------------------------------------------------------------
# Concrete shell (ADR-0012): one mechanism, the per-length coefficient
# (review pass 10, 10.12-10.14)
# ---------------------------------------------------------------------

_SHELL_DESCRIPTION = (
    "Per-length radial concrete-shell coefficient "
    "$C = \\rho_c/(2\\pi)\\,\\ln(r_b/r_a)$ in Ω·m (ADR-0012). Every "
    "segment of length $\\Delta s$ adds $C/\\Delta s$ to its reaction-matrix "
    "diagonal, the radial voltage drop through the shell; with the uniform "
    "current per electrode of the image family this adds exactly "
    "$C/L$ per ampere to the electrode potential ($L$: total wire length), "
    "the lumped shell resistance. ``wire_radius`` is the outer shell radius "
    "$r_b$ (the soil sees the shell surface). ``0.0`` (default): no shell. "
    "Honoured by the image family and ``solve_mutual_*``; the other "
    "backends reject a non-zero value."
)


def _shell_field() -> Any:
    """Field of ``concrete_shell_coefficient_ohm_m`` (same on every kind)."""
    return Field(
        default=0.0, ge=0.0, allow_inf_nan=False, description=_SHELL_DESCRIPTION
    )


def concrete_shell_coefficient(
    rho_c: float, wire_radius: float, thickness: float
) -> float:
    r"""Per-length coefficient of a cylindrical concrete shell (ADR-0012).

    .. math::

        C = \frac{\rho_c}{2\pi}\,\ln\frac{r_a + t}{r_a}

    Parameters
    ----------
    rho_c
        Resistivity of the concrete in Ω·m, ``>= 0``.
    wire_radius
        Radius :math:`r_a` of the bare wire in m, ``> 0``.
    thickness
        Radial thickness :math:`t` of the shell in m, ``>= 0``.

    Returns
    -------
    float
        :math:`C` in Ω·m; ``0.0`` for ``rho_c = 0`` or ``thickness = 0``.

    Raises
    ------
    ValueError
        If an argument is negative or not finite, or ``wire_radius`` is
        not positive.

    Notes
    -----
    The electrode then carries ``wire_radius = r_a + t`` and
    ``concrete_shell_coefficient_ohm_m = C``. For ``rho_c = 100`` Ω·m and
    5 cm of concrete around a 5 mm wire, :math:`C` = 38.16 Ω·m.
    """
    values = {"rho_c": rho_c, "wire_radius": wire_radius, "thickness": thickness}
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    if float(wire_radius) <= 0.0:
        raise ValueError(f"wire_radius must be positive (m), got {wire_radius!r}.")
    r_a = float(wire_radius)
    return float(rho_c) / (2.0 * math.pi) * math.log((r_a + float(thickness)) / r_a)


def shell_coefficient_from_resistance(
    electrode: "_ElectrodeBase", resistance_ohm: float
) -> float:
    r"""Coefficient that gives ``electrode`` a lumped shell resistance.

    The lumped shell model (ADR-0012 V1) describes the shell of one
    electrode by its total radial resistance :math:`R` in series with the
    soil. Spread uniformly over the electrode's wire, it becomes the
    per-length coefficient :math:`C = R\,L` with the total wire length
    :math:`L` (:meth:`~_ElectrodeBase.total_length`); the image family then
    adds exactly :math:`R` per ampere to the electrode potential. The
    generators use this conversion for ``concrete_model="lumped"``.

    Parameters
    ----------
    electrode
        Encased electrode (all of its wire is encased).
    resistance_ohm
        Lumped shell resistance :math:`R` in Ω, ``>= 0``.

    Returns
    -------
    float
        :math:`C` in Ω·m, for ``concrete_shell_coefficient_ohm_m``.

    Raises
    ------
    ValueError
        If ``resistance_ohm`` is negative or not finite.
    """
    resistance = float(resistance_ohm)
    if not math.isfinite(resistance) or resistance < 0.0:
        raise ValueError(
            f"resistance_ohm must be finite and non-negative, got {resistance_ohm!r}."
        )
    return resistance * electrode.total_length()


def shell_resistance_from_coefficient(
    electrode: "_ElectrodeBase", coefficient_ohm_m: float | None = None
) -> float:
    """Lumped shell resistance of ``electrode`` for a per-length coefficient.

    Inverse of :func:`shell_coefficient_from_resistance`:
    :math:`R = C / L` in Ω with the total wire length :math:`L` in m.

    Parameters
    ----------
    electrode
        Encased electrode.
    coefficient_ohm_m
        Coefficient :math:`C` in Ω·m; ``None`` (default) takes the
        electrode's own ``concrete_shell_coefficient_ohm_m``.

    Returns
    -------
    float
        :math:`R` in Ω.

    Raises
    ------
    ValueError
        If the coefficient is negative or not finite.
    """
    coefficient = float(
        electrode.concrete_shell_coefficient_ohm_m
        if coefficient_ohm_m is None
        else coefficient_ohm_m
    )
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError(
            f"coefficient_ohm_m must be finite and non-negative, got {coefficient!r}."
        )
    return coefficient / electrode.total_length()


class _ElectrodeBase(BaseModel):
    """Common base for all electrode geometries."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique name within the ``World``.")
    kind: str = Field(..., description="Discriminator for the geometry.")
    wire_radius: float = Field(
        default=0.005,
        gt=0.0,
        allow_inf_nan=False,
        description="Wire radius in m (default 5 mm ≙ ⌀ 10 mm).",
    )

    #: Kinds that carry ``concrete_shell_coefficient_ohm_m`` since 2026.11
    #: write it to JSON only when it is non-zero, so the JSON of every
    #: world without a shell stays byte-identical and loadable by 0.15.0.
    _SHELL_KEY_OPTIONAL: ClassVar[bool] = False

    @model_serializer(mode="wrap")
    def _omit_zero_shell(self, handler: Any) -> Any:
        """Drop a zero shell coefficient of the kinds that gained it in 2026.11."""
        data = handler(self)
        if (
            self._SHELL_KEY_OPTIONAL
            and isinstance(data, dict)
            and data.get("concrete_shell_coefficient_ohm_m") == 0.0
        ):
            del data["concrete_shell_coefficient_ohm_m"]
        return data

    @property
    def connection_point(self) -> Point3D:
        """Connection point used by incoming conductors.

        Subclasses override this. The default implementation raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Geometry queries (review pass 10, 10.79). Every electrode kind
    # answers them itself, so solvers, diagnostics, plots and exporters
    # no longer need an ``isinstance`` ladder over the kinds.
    # ------------------------------------------------------------------

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Discretise the electrode into straight segments.

        This is the discretisation every image-family backend uses (see
        :mod:`groundfield.geometry.primitives` for the conventions).

        Parameters
        ----------
        ds
            Target (maximum) segment length in m, ``> 0``.
        layer_interfaces
            Optional soil-layer interface depths in m (e.g. ``(h_1,)``).
            Wires whose depth range crosses an interface are split there
            first, so no segment straddles it. Horizontal electrodes are
            unaffected.

        Returns
        -------
        WireSegments
            Midpoints, lengths and unit tangents of the segments.
        """
        raise NotImplementedError

    def segment_count(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> int:
        """Number of segments :meth:`segments` produces for ``ds`` (in m)."""
        return len(self.segments(ds, layer_interfaces))

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Exact wire geometry as polylines for drawing and export.

        Parameters
        ----------
        n_arc
            Number of points used to sample a closed circular wire
            (first point repeated at the end). Straight wires are
            returned by their end points.

        Returns
        -------
        list[numpy.ndarray]
            One ``(K, 3)`` array of vertices in m per connected wire
            run.
        """
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box of the wire geometry.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(lower, upper)`` corners ``(x, y, z)`` in m, each of shape
            ``(3,)``.
        """
        pts = np.concatenate(self.wire_polylines(), axis=0)
        return pts.min(axis=0), pts.max(axis=0)

    def total_length(self) -> float:
        """Total wire length in m (all wires of the electrode)."""
        raise NotImplementedError

    def min_feature_length(self) -> float:
        """Smallest characteristic dimension in m.

        Compared with the segment length by the pre-flight resolution
        check: a feature shorter than ``segment_length`` is represented
        by the discretiser's lower floor only. Rod and strip: the
        length; ring: the circumference; polyline: the shortest edge;
        star: the arm length; mesh: the smallest of extent and spacing;
        grid mesh: the smaller mesh span.
        """
        raise NotImplementedError


class RodElectrode(_ElectrodeBase):
    """Vertical driven rod (Tiefenerder).

    Attributes
    ----------
    position
        Position of the head (top end) as ``(x, y, z)`` in metres.
        Typically the head sits just below the soil surface, e.g.
        ``z = 0.5``.
    length
        Rod length in metres (downwards along $+z$).
    concrete_shell_coefficient_ohm_m
        Per-length concrete-shell coefficient in Ω·m (ADR-0012, default
        0: no shell); see :func:`concrete_shell_coefficient`.
    """

    kind: Literal["rod"] = "rod"
    position: Point3D = Field(..., description="Head point (x, y, z) in m.")
    length: PositiveLength = Field(..., description="Rod length in m.")
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    _SHELL_KEY_OPTIONAL: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_dimensions(self) -> "RodElectrode":
        _require_longer_than_radius(self, self.length, "length")
        return self

    @property
    def connection_point(self) -> Point3D:
        return self.position

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Vertical rod: equal segments per soil layer (see base class).

        A part shorter than the wire diameter at an interface is not
        split off (see :data:`SLIVER_RADII`).
        """
        x0, y0, z0 = self.position
        head = np.array([x0, y0, 0.0])
        tangent = np.array([0.0, 0.0, 1.0])
        intervals = _merge_interface_slivers(
            _split_interval_at_interfaces(z0, z0 + self.length, layer_interfaces),
            SLIVER_RADII * self.wire_radius,
        )
        parts = []
        for z_lo, z_hi in intervals:
            part_len = z_hi - z_lo
            n = _segment_count(part_len, ds)
            parts.append(_straight_wire(head, tangent, z_lo, part_len / n, n))
        return WireSegments.concatenate(parts)

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Head and foot of the rod (see base class)."""
        x, y, z = self.position
        return [np.array([[x, y, z], [x, y, z + self.length]], dtype=float)]

    def total_length(self) -> float:
        """Rod length in m."""
        return float(self.length)

    def min_feature_length(self) -> float:
        """Rod length in m."""
        return float(self.length)


class RingElectrode(_ElectrodeBase):
    """Horizontal ring electrode (foundation / ring earth electrode).

    Attributes
    ----------
    center
        Ring centre ``(x, y, z)`` in metres. Typically buried 0.5 to
        1.0 m below the surface.
    radius
        Ring radius in metres.
    concrete_shell_coefficient_ohm_m
        Per-length concrete-shell coefficient in Ω·m (ADR-0012, default
        0: no shell); see :func:`concrete_shell_coefficient`.
    """

    kind: Literal["ring"] = "ring"
    center: Point3D = Field(..., description="Ring centre (x, y, z) in m.")
    radius: PositiveLength = Field(..., description="Ring radius in m.")
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    _SHELL_KEY_OPTIONAL: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_dimensions(self) -> "RingElectrode":
        _require_longer_than_radius(self, self.radius, "ring radius")
        return self

    @property
    def connection_point(self) -> Point3D:
        # Anchor at the first point of the ring (along +x).
        cx, cy, cz = self.center
        return (cx + self.radius, cy, cz)

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Horizontal ring: ``max(8, ceil(2πr/ds))`` equal arcs (see base class).

        Each arc is represented by its midpoint on the circle, its arc
        length and the tangent there.
        """
        perimeter = 2.0 * np.pi * self.radius
        n = _segment_count(perimeter, ds, minimum=8)
        seg_len = perimeter / n
        cx, cy, cz = self.center
        mids = np.empty((n, 3))
        dirs = np.empty((n, 3))
        for k in range(n):
            # Scalar trigonometry, kept from the historic discretiser so
            # the midpoints stay bit-identical across platforms.
            phi = 2.0 * np.pi * (k + 0.5) / n
            mids[k] = (
                cx + self.radius * np.cos(phi),
                cy + self.radius * np.sin(phi),
                cz,
            )
            dirs[k] = (-np.sin(phi), np.cos(phi), 0.0)
        return WireSegments(mids, np.full(n, seg_len), dirs)

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Closed circle sampled at ``n_arc`` points (see base class)."""
        cx, cy, cz = self.center
        theta = np.linspace(0.0, 2.0 * np.pi, n_arc)
        return [
            np.column_stack(
                [
                    cx + self.radius * np.cos(theta),
                    cy + self.radius * np.sin(theta),
                    np.full_like(theta, cz),
                ]
            )
        ]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Exact bounding box of the circle (see base class)."""
        cx, cy, cz = self.center
        r = self.radius
        return np.array([cx - r, cy - r, cz]), np.array([cx + r, cy + r, cz])

    def total_length(self) -> float:
        """Circumference ``2πr`` in m."""
        return float(2.0 * math.pi * self.radius)

    def min_feature_length(self) -> float:
        """Circumference ``2πr`` in m."""
        return float(2.0 * math.pi * self.radius)


class StripElectrode(_ElectrodeBase):
    """Horizontal straight strip earth electrode (Banderder).

    A buried straight wire from ``start`` to ``end``. Both end points
    must lie at the same depth ($z_\\text{start} = z_\\text{end}$);
    the in-plane direction is arbitrary, not restricted to the $x$
    or $y$ axis. The strip's length is the Euclidean distance between
    the end points.

    Attributes
    ----------
    start
        Strip start ``(x, y, z)`` in metres.
    end
        Strip end ``(x, y, z)`` in metres. Must have the same $z$ as
        ``start``.

    Notes
    -----
    The strip is the canonical *Banderder*. For the ``image`` /
    ``mom`` family the discretisation produces one chain of wire
    segments (no doubled-wire approximation). Plausibility tests
    compare against
    :func:`groundfield.references.dwight1936.horizontal_wire`.
    """

    kind: Literal["strip"] = "strip"
    start: Point3D = Field(..., description="Strip start (x, y, z) in m.")
    end: Point3D = Field(..., description="Strip end (x, y, z) in m.")
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    @model_validator(mode="after")
    def _check_horizontal(self) -> "StripElectrode":
        if abs(self.start[2] - self.end[2]) > 1e-9:
            raise ValueError(
                "StripElectrode is horizontal: start[2] must equal end[2]. "
                f"Got start={self.start}, end={self.end}."
            )
        # Zero-length strips used to reach the solver and fail there as a
        # thin-wire violation of a 0 m segment (review pass 10, 10.67).
        if self.length <= _MIN_EDGE_LENGTH_M:
            raise ValueError(
                f"StripElectrode {self.name!r} has (near) zero length: start "
                f"{self.start} and end {self.end} coincide."
            )
        _require_longer_than_radius(self, self.length, "length")
        return self

    @property
    def length(self) -> float:
        """Euclidean strip length in metres."""
        sx, sy, _ = self.start
        ex, ey, _ = self.end
        return float(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)

    @property
    def connection_point(self) -> Point3D:
        return self.start

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Straight strip: equal segments along the axis (see base class).

        A sloped strip whose depth range crosses a layer interface would
        be split there first (ADR-0007 step 1); the model validator
        keeps strips horizontal, so this only guards the invariant.
        """
        L = self.length
        p0 = np.array(self.start, dtype=float)
        p1 = np.array(self.end, dtype=float)
        direction = (p1 - p0) / L
        # Map interface depths to axis parameters t in [0, L].
        t_cuts: list[float] = []
        dz = p1[2] - p0[2]
        if layer_interfaces and abs(dz) > 1e-12:
            for h in layer_interfaces:
                t_cuts.append((h - p0[2]) / dz * L)
        intervals = _merge_interface_slivers(
            _split_interval_at_interfaces(0.0, L, t_cuts),
            SLIVER_RADII * self.wire_radius,
        )
        parts = []
        for t_lo, t_hi in intervals:
            part_len = t_hi - t_lo
            n = _segment_count(part_len, ds)
            parts.append(_straight_wire(p0, direction, t_lo, part_len / n, n))
        return WireSegments.concatenate(parts)

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Start and end point (see base class)."""
        return [np.array([list(self.start), list(self.end)], dtype=float)]

    def total_length(self) -> float:
        """Strip length in m."""
        return float(self.length)

    def min_feature_length(self) -> float:
        """Strip length in m."""
        return float(self.length)


class PolylineElectrode(_ElectrodeBase):
    """Buried wire following an arbitrary horizontal polyline.

    Generalises :class:`StripElectrode` from a single straight wire to
    a chain of straight wires through ``vertices``. With
    ``closed = True`` the last vertex is wired back to the first,
    which makes this the *Ringerder along an arbitrary building
    outline* — the DIN-18014 Streifenfundament that follows the real
    building perimeter instead of its bounding rectangle.

    All vertices must lie at the same depth (the wire is horizontal),
    consistent with :class:`StripElectrode`.

    Attributes
    ----------
    vertices
        Ordered polyline vertices ``(x, y, z)`` in metres, at least
        two. For ``closed = True`` do **not** repeat the first vertex
        at the end — the closing edge is implicit.
    closed
        Whether the polyline is closed into a ring (default
        ``True``).

    Notes
    -----
    Motivation (AP1): approximating a building footprint by its
    oriented bounding rectangle (OMBR) overestimates the electrode
    perimeter for L- and U-shaped buildings and — worse — lets the
    rectangles of neighbouring buildings coincide or overlap, which
    drives distinct segments into the solver's 1 mm singularity clamp.
    Following the real polygon removes both artefacts by construction.
    """

    kind: Literal["polyline"] = "polyline"
    vertices: list[Point3D] = Field(
        ...,
        min_length=2,
        description="Polyline vertices (x, y, z) in m, same depth.",
    )
    closed: bool = Field(
        default=True,
        description="Close the polyline into a ring (last vertex wired to first).",
    )
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    @model_validator(mode="after")
    def _check_horizontal_and_nondegenerate(self) -> "PolylineElectrode":
        z0 = self.vertices[0][2]
        if any(abs(v[2] - z0) > 1e-9 for v in self.vertices):
            raise ValueError(
                "PolylineElectrode is horizontal: every vertex must share "
                f"the same z. Got {self.vertices}."
            )
        # Reject degenerate (zero-length) edges. A consecutive duplicate
        # vertex — or, for ``closed=True``, repeating the first vertex at
        # the end — collapses one segment to zero length. Left unchecked
        # the discretiser absorbs it silently and the reaction matrix
        # carries a spurious pair of coincident segments (driven into the
        # solver's singularity clamp). ``self.length > 0`` does not catch
        # this: only the individual edge is degenerate.
        n = len(self.vertices)
        for k, ((sx, sy, _), (ex, ey, _)) in enumerate(self.edges):
            if ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5 <= _MIN_EDGE_LENGTH_M:
                a, b = self.vertices[k], self.vertices[(k + 1) % n]
                raise ValueError(
                    f"PolylineElectrode edge {k} has (near) zero length: "
                    f"vertices {a} and {b} coincide. Remove duplicate "
                    "vertices; for ``closed=True`` do not repeat the first "
                    "vertex at the end (the closing edge is implicit)."
                )
        if self.closed and n == 2:
            warnings.warn(
                "PolylineElectrode(closed=True) needs at least three "
                "vertices to form a ring; with two vertices it is treated "
                "as an open wire (no closing edge is added).",
                UserWarning,
                stacklevel=2,
            )
        _require_longer_than_radius(self, self.min_feature_length(), "shortest edge")
        return self

    @property
    def edges(self) -> list[tuple[Point3D, Point3D]]:
        """Straight wires making up the polyline (closing edge included)."""
        vs = self.vertices
        out = [(vs[k], vs[k + 1]) for k in range(len(vs) - 1)]
        if self.closed and len(vs) > 2:
            out.append((vs[-1], vs[0]))
        return out

    @property
    def length(self) -> float:
        """Total wire length in metres (sum over all edges)."""
        total = 0.0
        for (sx, sy, _), (ex, ey, _) in self.edges:
            total += float(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)
        return total

    @property
    def connection_point(self) -> Point3D:
        return self.vertices[0]

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """One strip-style chain per edge (see base class).

        Every edge is discretised exactly like a :class:`StripElectrode`,
        so a closed rectangular polyline reproduces the four perimeter
        wires of the equivalent strip chain. Degenerate edges (zero
        length) are skipped.
        """
        return _edge_wires(self.edges, ds, skip_degenerate=True)

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """The vertex chain, closed if ``closed`` (see base class)."""
        pts = [list(v) for v in self.vertices]
        if self.closed and len(pts) > 2:
            pts.append(list(self.vertices[0]))
        return [np.array(pts, dtype=float)]

    def total_length(self) -> float:
        """Sum of the edge lengths in m."""
        return float(self.length)

    def min_feature_length(self) -> float:
        """Shortest edge in m."""
        return float(
            min(
                ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                for (sx, sy, _), (ex, ey, _) in self.edges
            )
        )


class StarElectrode(_ElectrodeBase):
    r"""Buried horizontal star electrode (*Sternerder*).

    ``n_arms`` equal-length wires radiate from a common centre node at a
    single depth, spaced uniformly by $360^\circ / n$. This is the
    classic $n$-point star of Dwight (1936, Eqs. 23–26); the closed-form
    grounding resistance is available as
    :func:`groundfield.references.dwight1936.n_point_star` for
    validation (Dwight covers 3, 4, 6 and 8 arms).

    Attributes
    ----------
    center
        Hub ``(x, y, z)`` in metres. Every arm shares this point and
        stays at ``z = center[2]`` — the wire is horizontal.
    n_arms
        Number of radial arms ($\ge 2$).
    arm_length
        Length $L$ of a single arm in metres.
    orientation_deg
        Azimuth of the first arm measured from the world ``+x`` axis in
        degrees; the remaining arms follow at multiples of
        $360^\circ / n$.
    concrete_shell_coefficient_ohm_m
        Optional per-segment radial concrete-shell coefficient, same
        semantics as on :class:`StripElectrode` (ADR-0012 V2).

    Notes
    -----
    Discretisation and the reaction-matrix assembly treat the star as
    ``n_arms`` straight wires sharing the centre node, so it forms a
    single galvanic cluster — exactly the physical *Sternerder*.
    """

    kind: Literal["star"] = "star"
    center: Point3D = Field(
        ...,
        description="Hub (x, y, z) in m; every arm stays at z = center[2].",
    )
    n_arms: int = Field(..., ge=2, description="Number of radial arms (>= 2).")
    arm_length: PositiveLength = Field(
        ..., description="Length L of a single arm in m."
    )
    orientation_deg: FiniteFloat = Field(
        default=0.0,
        description="Azimuth of the first arm from +x in degrees.",
    )
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    @model_validator(mode="after")
    def _check_dimensions(self) -> "StarElectrode":
        _require_longer_than_radius(self, self.arm_length, "arm length")
        return self

    @property
    def arm_tips(self) -> list[Point3D]:
        """Outer endpoint ``(x, y, z)`` of each arm."""
        cx, cy, cz = self.center
        tips: list[Point3D] = []
        for k in range(self.n_arms):
            theta = math.radians(self.orientation_deg + k * 360.0 / self.n_arms)
            tips.append(
                (
                    cx + self.arm_length * math.cos(theta),
                    cy + self.arm_length * math.sin(theta),
                    cz,
                )
            )
        return tips

    @property
    def edges(self) -> list[tuple[Point3D, Point3D]]:
        """Straight wires making up the star (centre → each tip)."""
        return [(self.center, tip) for tip in self.arm_tips]

    @property
    def length(self) -> float:
        """Total wire length in metres (``n_arms * arm_length``)."""
        return float(self.n_arms * self.arm_length)

    @property
    def connection_point(self) -> Point3D:
        return self.center

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """One strip-style chain per arm, centre to tip (see base class)."""
        return _edge_wires(self.edges, ds, skip_degenerate=False)

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """One straight wire per arm (see base class)."""
        return [np.array([list(a), list(b)], dtype=float) for a, b in self.edges]

    def total_length(self) -> float:
        """``n_arms * arm_length`` in m."""
        return float(self.length)

    def min_feature_length(self) -> float:
        """Arm length in m."""
        return float(self.arm_length)


class MeshElectrode(_ElectrodeBase):
    """Rectangular mesh earth electrode (uniform spacing).

    Backwards-compatible primitive that sets the inner mesh density
    via a single ``spacing`` parameter. For finer control over the
    column / row count, prefer :class:`GridMeshElectrode`.

    Attributes
    ----------
    corner
        Corner point ``(x, y, z)`` in metres (min-x, min-y, depth).
    size
        Extent ``(dx, dy)`` of the mesh in metres.
    spacing
        Spacing between adjacent longitudinal / transverse wires in m.
    concrete_shell_coefficient_ohm_m
        Per-length concrete-shell coefficient in Ω·m (ADR-0012, default
        0: no shell); see :func:`concrete_shell_coefficient`.
    """

    kind: Literal["mesh"] = "mesh"
    corner: Point3D = Field(..., description="Corner (x, y, z) in m.")
    size: tuple[PositiveLength, PositiveLength] = Field(
        ..., description="Extent (dx, dy) in m, both > 0."
    )
    spacing: PositiveLength = Field(default=1.0, description="Mesh spacing in m.")
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    _SHELL_KEY_OPTIONAL: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_dimensions(self) -> "MeshElectrode":
        _require_longer_than_radius(
            self, self.min_feature_length(), "smallest extent or spacing"
        )
        return self

    @property
    def connection_point(self) -> Point3D:
        # Anchor at the centre of the mesh.
        cx, cy, cz = self.corner
        dx, dy = self.size
        return (cx + dx / 2.0, cy + dy / 2.0, cz)

    def _wire_counts(self) -> tuple[int, int]:
        """Wire positions ``(n_x_positions, n_y_positions)`` from ``spacing``."""
        dx, dy = self.size
        nx = max(2, int(np.round(dx / self.spacing)) + 1)
        ny = max(2, int(np.round(dy / self.spacing)) + 1)
        return nx, ny

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Grid of wires at ``spacing`` (see base class)."""
        cx, cy, cz = self.corner
        dx, dy = self.size
        nx, ny = self._wire_counts()
        return _grid_wires(
            cx=cx, cy=cy, cz=cz, dx=dx, dy=dy, nx_wires=ny, ny_wires=nx, ds=ds
        )

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Every wire of the mesh (see base class)."""
        cx, cy, cz = self.corner
        dx, dy = self.size
        nx, ny = self._wire_counts()
        return _grid_wire_polylines(
            cx=cx, cy=cy, cz=cz, dx=dx, dy=dy, nx_wires=ny, ny_wires=nx
        )

    def total_length(self) -> float:
        """Length of all wires in m."""
        dx, dy = self.size
        nx, ny = self._wire_counts()
        return float(ny * dx + nx * dy)

    def min_feature_length(self) -> float:
        """Smallest of the two extents and the spacing, in m."""
        return float(min(self.size[0], self.size[1], self.spacing))


class GridMeshElectrode(_ElectrodeBase):
    """Rectangular meshed earth electrode with explicit n × m divisions.

    Defines an axis-aligned rectangle of size ``(dx, dy)`` at depth
    ``corner[2]``, divided into ``n_x`` × ``n_y`` internal meshes.
    The geometry consists of ``n_x + 1`` longitudinal wires (running
    along the local $x$ axis from ``corner`` to ``corner + (dx, 0)``)
    and ``n_y + 1`` transverse wires (running along the local $y$
    axis). All wires share the same depth.

    Attributes
    ----------
    corner
        Corner point ``(x, y, z)`` in metres (min-x, min-y, depth).
    size
        Extent ``(dx, dy)`` of the rectangle in metres.
    n_x
        Number of meshes along $x$ (must be ≥ 1). The number of
        longitudinal wires is ``n_y + 1``.
    n_y
        Number of meshes along $y$ (must be ≥ 1). The number of
        transverse wires is ``n_x + 1``.
    concrete_shell_coefficient_ohm_m
        Per-length concrete-shell coefficient in Ω·m (ADR-0012, default
        0: no shell); see :func:`concrete_shell_coefficient`.

    Notes
    -----
    Naming convention: ``n_x`` × ``n_y`` is the count of *meshes*
    (cells), not of wires.
    A grid with ``n_x = n_y = 1`` is therefore a single closed
    rectangle (the perimeter wires only).
    """

    kind: Literal["grid_mesh"] = "grid_mesh"
    corner: Point3D = Field(..., description="Corner (x, y, z) in m.")
    size: tuple[PositiveLength, PositiveLength] = Field(
        ..., description="Extent (dx, dy) in m, both > 0."
    )
    n_x: int = Field(..., ge=1, description="Number of meshes along x (≥ 1).")
    n_y: int = Field(..., ge=1, description="Number of meshes along y (≥ 1).")
    concrete_shell_coefficient_ohm_m: float = _shell_field()

    _SHELL_KEY_OPTIONAL: ClassVar[bool] = True

    @model_validator(mode="after")
    def _check_dimensions(self) -> "GridMeshElectrode":
        _require_longer_than_radius(
            self, self.min_feature_length(), "smaller mesh span"
        )
        return self

    @property
    def connection_point(self) -> Point3D:
        cx, cy, cz = self.corner
        dx, dy = self.size
        return (cx + dx / 2.0, cy + dy / 2.0, cz)

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Grid of ``n_y + 1`` by ``n_x + 1`` wires (see base class)."""
        cx, cy, cz = self.corner
        dx, dy = self.size
        return _grid_wires(
            cx=cx,
            cy=cy,
            cz=cz,
            dx=dx,
            dy=dy,
            nx_wires=self.n_y + 1,
            ny_wires=self.n_x + 1,
            ds=ds,
        )

    def wire_polylines(self, n_arc: int = 64) -> list[np.ndarray]:
        """Every wire of the grid (see base class)."""
        cx, cy, cz = self.corner
        dx, dy = self.size
        return _grid_wire_polylines(
            cx=cx,
            cy=cy,
            cz=cz,
            dx=dx,
            dy=dy,
            nx_wires=self.n_y + 1,
            ny_wires=self.n_x + 1,
        )

    def total_length(self) -> float:
        """Length of all wires in m."""
        dx, dy = self.size
        return float((self.n_y + 1) * dx + (self.n_x + 1) * dy)

    def min_feature_length(self) -> float:
        """Smaller mesh span in m."""
        return float(min(self.size[0] / self.n_x, self.size[1] / self.n_y))


# Discriminated union used inside ``World``.
Electrode = Union[
    RodElectrode,
    RingElectrode,
    StripElectrode,
    PolylineElectrode,
    StarElectrode,
    MeshElectrode,
    GridMeshElectrode,
]
