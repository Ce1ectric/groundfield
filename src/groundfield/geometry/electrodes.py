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
from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundfield.geometry.primitives import (
    WireSegments,
    _edge_wires,
    _grid_wire_polylines,
    _grid_wires,
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
]

# 3-D coordinate (x, y, z); z axis points into the soil.
Point3D = tuple[float, float, float]

#: Edges shorter than this (metres) are treated as degenerate: a
#: :class:`PolylineElectrode` rejects them so a duplicate vertex cannot
#: silently collapse a segment to zero length. Set far below any physical
#: wire radius, so only true coincidences trip it.
_MIN_EDGE_LENGTH_M = 1e-9


class _ElectrodeBase(BaseModel):
    """Common base for all electrode geometries."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Unique name within the ``World``.")
    kind: str = Field(..., description="Discriminator for the geometry.")
    wire_radius: float = Field(
        default=0.005,
        gt=0.0,
        description="Wire radius in m (default 5 mm ≙ ⌀ 10 mm).",
    )

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
    """

    kind: Literal["rod"] = "rod"
    position: Point3D = Field(..., description="Head point (x, y, z) in m.")
    length: float = Field(..., gt=0.0, description="Rod length in m.")

    @property
    def connection_point(self) -> Point3D:
        return self.position

    def segments(
        self, ds: float, layer_interfaces: Sequence[float] | None = None
    ) -> WireSegments:
        """Vertical rod: equal segments per soil layer (see base class)."""
        x0, y0, z0 = self.position
        head = np.array([x0, y0, 0.0])
        tangent = np.array([0.0, 0.0, 1.0])
        parts = []
        for z_lo, z_hi in _split_interval_at_interfaces(
            z0, z0 + self.length, layer_interfaces
        ):
            part_len = z_hi - z_lo
            n = max(1, int(np.ceil(part_len / ds)))
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
    """

    kind: Literal["ring"] = "ring"
    center: Point3D = Field(..., description="Ring centre (x, y, z) in m.")
    radius: float = Field(..., gt=0.0, description="Ring radius in m.")

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
        n = max(8, int(np.ceil(perimeter / ds)))
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
    concrete_shell_coefficient_ohm_m: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-segment radial concrete-shell coefficient "
            "$C = \\rho_c/(2\\pi)\\,\\ln(r_b/r_a)$ in Ω·m. "
            "``0.0`` (default) — no shell, behaves as before. "
            "When positive, the MoM diagonal entry of every "
            "segment of length $\\Delta s$ in this strip is "
            "augmented by $C/\\Delta s$ — the radial voltage drop "
            "through the cylindrical shell (ADR-0012 V2 path). "
            "Currently honoured by the ``image`` and "
            "``image_2layer`` backends; the others raise "
            "``NotImplementedError`` so a misconfiguration is "
            "loud rather than silently wrong."
        ),
    )

    @model_validator(mode="after")
    def _check_horizontal(self) -> "StripElectrode":
        if abs(self.start[2] - self.end[2]) > 1e-9:
            raise ValueError(
                "StripElectrode is horizontal: start[2] must equal end[2]. "
                f"Got start={self.start}, end={self.end}."
            )
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
        parts = []
        for t_lo, t_hi in _split_interval_at_interfaces(0.0, L, t_cuts):
            part_len = t_hi - t_lo
            n = max(1, int(np.ceil(part_len / ds)))
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
    concrete_shell_coefficient_ohm_m: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-segment radial concrete-shell coefficient "
            "$C = \\rho_c/(2\\pi)\\,\\ln(r_b/r_a)$ in Ω·m — same "
            "semantics as on :class:`StripElectrode` (ADR-0012 V2)."
        ),
    )

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
    arm_length: float = Field(..., gt=0.0, description="Length L of a single arm in m.")
    orientation_deg: float = Field(
        default=0.0,
        description="Azimuth of the first arm from +x in degrees.",
    )
    concrete_shell_coefficient_ohm_m: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Per-segment radial concrete-shell coefficient "
            "$C = \\rho_c/(2\\pi)\\,\\ln(r_b/r_a)$ in Ω·m — same "
            "semantics as on :class:`StripElectrode` (ADR-0012 V2)."
        ),
    )

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
    """

    kind: Literal["mesh"] = "mesh"
    corner: Point3D = Field(..., description="Corner (x, y, z) in m.")
    size: tuple[float, float] = Field(..., description="Extent (dx, dy) in m.")
    spacing: float = Field(default=1.0, gt=0.0, description="Mesh spacing in m.")

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

    Notes
    -----
    Naming convention: ``n_x`` × ``n_y`` is the count of *meshes*
    (cells), not of wires.
    A grid with ``n_x = n_y = 1`` is therefore a single closed
    rectangle (the perimeter wires only).
    """

    kind: Literal["grid_mesh"] = "grid_mesh"
    corner: Point3D = Field(..., description="Corner (x, y, z) in m.")
    size: tuple[float, float] = Field(..., description="Extent (dx, dy) in m.")
    n_x: int = Field(..., ge=1, description="Number of meshes along x (≥ 1).")
    n_y: int = Field(..., ge=1, description="Number of meshes along y (≥ 1).")

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
