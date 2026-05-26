r"""PEN backbone topologies for the TN low-voltage generator.

The :class:`~groundfield.generators.tn_network.TnNetworkGenerator`
historically wires its PEN backbone as a *star* of cable-cabinets
around the substation: every cable cabinet (KVS) is connected to the
substation directly and every building connects to its *nearest* KVS
by Manhattan distance. That topology is what the older Audit-pass
fixtures still exercise — it stays available as the default through
:class:`StarKvsTopology`.

This module adds a second topology that matches the typical layout
of a German :emphasis:`TN-Ortsnetz`:

* the substation feeds **N radial feeders** (LV cables), each one
  emanating along a discrete direction (north / east / south / west
  by default);
* each feeder has a **finite slot budget** for direct building taps
  at the substation. When the slot budget is exhausted, the feeder
  is *extended* by inserting a KVS along its axis. Each KVS comes
  with its own slot budget for the next batch of buildings;
* buildings are assigned to the **angularly closest** feeder, then
  to the **distance-closest** tap point along that feeder that still
  has a free slot;
* the feeder is :emphasis:`Manhattan-routed` in the sense that the
  trunk runs along a single axis (the feeder's discrete direction)
  and KVSes are placed on that axis. The straight-line conductor
  between two trunk anchors is therefore already axis-aligned.
  Building service drops connect the corresponding tap anchor to
  the building's foundation centre by a single straight line.

Physical interpretation
-----------------------
A real LV radial cable from the substation to the last cabinet is
typically up to about 500 m long below 1 kHz the quasi-static
formulation stays valid, and the cable's series-resistance
$R_\text{ser} = \rho L / A$ together with the optional Neumann
inductance (ADR-0004) reproduces the dominant longitudinal drop. The
finite slot count per source reflects the fact that real cabinets
distribute the load along the street so that no single connection
point carries more than a handful of houses.

Limitations
-----------
* The trunk between two anchors is a single straight conductor in
  3D. The :emphasis:`Manhattan-corner` rerouting of service drops
  (tap → corner → building) is a v2 enhancement. For typical
  street-aligned plots the corner offset is small compared with the
  full feeder length and the residual error in the longitudinal
  branch impedance stays well below 1 % at 50 Hz.
* When :attr:`RadialTrunkTopology` is selected, the parent
  :class:`groundfield.generators.tn_network.KvsConfig.placement`
  and :attr:`~KvsConfig.fixed_count` / ``quote_per_100_buildings``
  fields are ignored — the topology decides KVS count and
  positions. The KVS *grounding* spec from ``KvsConfig.grounding``
  is still honoured (so the user keeps a single, central place to
  configure cabinet earthing).
"""

from __future__ import annotations

import math
import warnings
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "StarKvsTopology",
    "RadialTrunkTopology",
    "PenTopology",
]


# ---------------------------------------------------------------------
# Star-KVS topology (default; legacy behaviour)
# ---------------------------------------------------------------------


class StarKvsTopology(BaseModel):
    """Legacy *star* PEN topology.

    Substation feeds each KVS directly, every building taps to its
    nearest KVS (Manhattan metric). KVS placement and count come from
    :class:`groundfield.generators.tn_network.KvsConfig`.

    This is the default for backward compatibility — pre-v0.7
    generator configurations behave bit-identically.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["star_kvs"] = "star_kvs"


# ---------------------------------------------------------------------
# Radial-trunk topology (new)
# ---------------------------------------------------------------------


class RadialTrunkTopology(BaseModel):
    r"""Radial-feeder PEN topology with finite per-source slot budget.

    The substation feeds :attr:`n_feeders` radial feeders. Each
    feeder runs along its own discrete direction (:attr:`feeder_directions_deg`
    or — by default — evenly spaced around the substation starting
    at 0°). Buildings are assigned to the angularly closest feeder,
    then sorted by distance from the substation along that feeder
    axis. The first :attr:`slots_per_substation` buildings tap
    directly to the substation; once the budget is exhausted, a KVS
    is inserted along the feeder axis and the next
    :attr:`slots_per_kvs` buildings tap to that KVS, and so on.

    Attributes
    ----------
    n_feeders
        Number of radial feeders emanating from the substation.
        Default 4 (a typical "north / east / south / west"
        TN-Ortsnetz).
    feeder_directions_deg
        Optional explicit feeder angles in degrees (CCW from +x).
        Length must equal :attr:`n_feeders`. ``None`` (default)
        auto-distributes the feeders evenly:
        :math:`\theta_k = 360^\circ\,k/N` for :math:`k=0,\ldots,N-1`.
    max_feeder_length_m
        Maximum trunk length per feeder. Buildings beyond this
        distance (measured along the feeder axis) are dropped from
        the assignment and trigger a :class:`UserWarning`.
    slots_per_substation
        Number of direct building taps each feeder may have at the
        substation. Once this many buildings have been wired, the
        next building forces a KVS extension.
    slots_per_kvs
        Number of direct building taps per KVS. After the substation
        budget is exhausted, KVSes are inserted along the feeder
        until every assigned building is wired or the feeder reaches
        :attr:`max_feeder_length_m`.
    kvs_spacing_m
        Step length (along the feeder axis) between adjacent KVSes
        when the topology decides to insert a new one. Default 60 m
        matches the typical street-section length between cable
        cabinets in urban LV networks.
    name_prefix
        Prefix used to name the inserted KVSes
        (``f"{prefix}_{feeder_idx}_{kvs_idx}"``). Defaults to ``"kvs"``
        so the names are consistent with the legacy star topology.

    Notes
    -----
    Trunk segments (substation → KVS and KVS → KVS) and service
    drops (tap anchor → building anchor) are realised as straight
    3D conductors between the corresponding anchor electrodes. The
    Manhattan-ness of the layout is reflected in (a) the discrete
    feeder directions which keep the trunk axis-aligned and (b) the
    angular feeder assignment, which steers buildings toward the
    geometrically closest street axis. Corner-routed service drops
    are tracked as a v2 enhancement.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["radial_trunk"] = "radial_trunk"

    n_feeders: int = Field(
        default=4, ge=1,
        description="Number of radial feeders out of the substation.",
    )
    feeder_directions_deg: Optional[list[float]] = Field(
        default=None,
        description=(
            "Optional explicit feeder angles in degrees (CCW from "
            "+x). Length must equal ``n_feeders``. ``None`` auto-"
            "distributes the feeders evenly."
        ),
    )
    max_feeder_length_m: float = Field(
        default=500.0, gt=0.0,
        description=(
            "Maximum trunk length per feeder in m. Buildings further "
            "out are dropped with a UserWarning."
        ),
    )
    slots_per_substation: int = Field(
        default=8, ge=1,
        description=(
            "Number of direct building taps per feeder at the "
            "substation."
        ),
    )
    slots_per_kvs: int = Field(
        default=8, ge=1,
        description="Number of direct building taps per inserted KVS.",
    )
    kvs_spacing_m: float = Field(
        default=60.0, gt=0.0,
        description=(
            "Axial step length between consecutive KVSes on a feeder "
            "in m. Typical 50–80 m in urban LV networks."
        ),
    )
    name_prefix: str = Field(
        default="kvs",
        description="Name prefix for the inserted KVSes.",
    )

    @field_validator("feeder_directions_deg")
    @classmethod
    def _validate_directions(
        cls, value: Optional[list[float]],
    ) -> Optional[list[float]]:
        if value is None:
            return value
        if not all(math.isfinite(float(v)) for v in value):
            raise ValueError(
                "feeder_directions_deg must contain finite floats."
            )
        return [float(v) for v in value]

    # -----------------------------------------------------------------
    # Helpers consumed by the generator
    # -----------------------------------------------------------------

    def resolved_directions_rad(self) -> list[float]:
        """Return the feeder directions in radians.

        If :attr:`feeder_directions_deg` is ``None``, the directions
        are auto-distributed evenly around the substation. Otherwise
        the explicit list is consumed verbatim (after the
        :func:`field_validator` check that the length matches
        :attr:`n_feeders`).
        """
        if self.feeder_directions_deg is None:
            return [
                2.0 * math.pi * k / self.n_feeders
                for k in range(self.n_feeders)
            ]
        if len(self.feeder_directions_deg) != self.n_feeders:
            raise ValueError(
                "feeder_directions_deg length "
                f"({len(self.feeder_directions_deg)}) does not match "
                f"n_feeders ({self.n_feeders})."
            )
        return [math.radians(a) for a in self.feeder_directions_deg]

    def assign_buildings_to_feeders(
        self,
        substation_xy: tuple[float, float],
        building_positions: list[tuple[float, float]],
    ) -> list[list[int]]:
        """Group building indices by angularly closest feeder.

        Parameters
        ----------
        substation_xy
            Substation centre in m.
        building_positions
            Per-building centres in m. Order is preserved across
            the assignment.

        Returns
        -------
        list of lists of int
            ``out[feeder_idx]`` is the list of building indices
            assigned to feeder ``feeder_idx``. Each list is sorted
            by distance from the substation along the feeder axis
            (signed projection onto the feeder direction). Buildings
            with a *negative* axial projection (behind the
            substation) are excluded from the feeder.
        """
        sx, sy = substation_xy
        feeder_dirs = self.resolved_directions_rad()
        groups: list[list[tuple[float, int]]] = [[] for _ in feeder_dirs]
        for i, (bx, by) in enumerate(building_positions):
            dx, dy = bx - sx, by - sy
            r = math.hypot(dx, dy)
            if r < 1e-9:
                # Building on the substation; assign to first feeder.
                groups[0].append((0.0, i))
                continue
            building_angle = math.atan2(dy, dx)
            # Pick feeder with smallest angular distance.
            best = min(
                range(len(feeder_dirs)),
                key=lambda k: _angle_diff(feeder_dirs[k], building_angle),
            )
            # Signed projection onto feeder direction.
            theta = feeder_dirs[best]
            proj = dx * math.cos(theta) + dy * math.sin(theta)
            if proj <= 0.0:
                # Behind the substation along this feeder axis;
                # cannot be reached without back-tracking. Drop.
                warnings.warn(
                    f"RadialTrunkTopology: building at ({bx:.1f}, "
                    f"{by:.1f}) projects behind the closest feeder "
                    f"axis (theta={math.degrees(theta):.0f}°, "
                    f"proj={proj:.2f} m); dropping from the PEN "
                    "topology.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if proj > self.max_feeder_length_m:
                warnings.warn(
                    f"RadialTrunkTopology: building at ({bx:.1f}, "
                    f"{by:.1f}) projects to {proj:.1f} m on its "
                    "feeder axis, beyond max_feeder_length_m="
                    f"{self.max_feeder_length_m} m; dropping from "
                    "the PEN topology.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            groups[best].append((proj, i))
        # Sort each group by axial distance from the substation.
        out: list[list[int]] = []
        for g in groups:
            g.sort(key=lambda pair: pair[0])
            out.append([i for _, i in g])
        return out

    def plan_feeder_kvs_positions(
        self,
        substation_xy: tuple[float, float],
        feeder_idx: int,
        n_buildings_on_feeder: int,
    ) -> list[tuple[float, float]]:
        r"""Compute KVS positions for a feeder given its building count.

        The first :attr:`slots_per_substation` buildings tap directly
        to the substation. The next batch of up to
        :attr:`slots_per_kvs` buildings shares one KVS, the batch
        after that shares the next KVS, and so on. Each KVS sits on
        the feeder axis at the smallest multiple of
        :attr:`kvs_spacing_m` that is :math:`\ge` the projection of
        the *first* building it serves — but we don't have that
        projection here; we approximate by stepping one
        :attr:`kvs_spacing_m` per KVS. This keeps the layout
        predictable (KVSes are equispaced along the street) and
        matches typical Ortsnetz cable cabinet spacing.

        Positions beyond :attr:`max_feeder_length_m` are clipped to
        :attr:`max_feeder_length_m`; if multiple KVSes would be
        clipped to the same maximum distance, only the first is
        kept and the remaining buildings will trigger a
        :class:`UserWarning` from the caller.
        """
        sx, sy = substation_xy
        theta = self.resolved_directions_rad()[feeder_idx]
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        remaining = max(0, n_buildings_on_feeder - self.slots_per_substation)
        if remaining <= 0:
            return []
        n_kvs = math.ceil(remaining / self.slots_per_kvs)
        positions: list[tuple[float, float]] = []
        last_distance: Optional[float] = None
        for k in range(n_kvs):
            distance = (k + 1) * self.kvs_spacing_m
            if distance > self.max_feeder_length_m:
                if last_distance is not None and last_distance >= (
                    self.max_feeder_length_m - 1e-9
                ):
                    break
                distance = self.max_feeder_length_m
            kvs_xy = (sx + distance * cos_t, sy + distance * sin_t)
            positions.append(kvs_xy)
            last_distance = distance
        return positions


# ---------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------


PenTopology = Annotated[
    Union[StarKvsTopology, RadialTrunkTopology],
    Field(discriminator="kind"),
]
"""JSON-serialisable union of PEN topologies.

Add new topology classes here when a future Ortsnetz layout
(meshed-LV, ring, mixed) is implemented; the
:class:`~groundfield.generators.tn_network.TnNetworkGenerator`
dispatches on :attr:`kind`.
"""


# ---------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference in $[0, \\pi]$."""
    d = (a - b) % (2.0 * math.pi)
    if d > math.pi:
        d = 2.0 * math.pi - d
    return d
