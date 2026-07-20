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
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
