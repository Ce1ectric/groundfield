"""Composable grounding-system specifications.

A :class:`GroundingSystemSpec` represents *one* electrically
connected grounding installation: a (possibly heterogeneous) set of
:class:`~groundfield.generators.electrode_specs.ElectrodeSpec`
instances that together form a single galvanic cluster. The class
is shared between

* the substation grounding (typically ring + multiple rods, optionally
  a strip and/or a foundation electrode),
* every cable cabinet (KVS) grounding (typically a single rod, but
  any combination is allowed),
* the per-house / per-building grounding (foundation, rod, mesh, or
  any AND-combination).

The materialisation method :meth:`GroundingSystemSpec.build_at`
takes a :class:`World`, a centre position ``(x, y)``, and a name
prefix; for each electrode in the list it draws a Bernoulli on
``presence_prob``, and if present, samples the geometry parameters
and registers the corresponding electrode with the world. All
created electrodes are then bonded together with bare-copper
conductors so that the world treats them as one cluster.

The first present electrode is the *anchor* — the
:meth:`build_at` method returns its name so the caller can wire
PEN cables, current sources, etc. into the cluster.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from pydantic import Field

from groundfield.api import create_conductor, create_electrode
from groundfield.generators.base import GeneratorConfig
from groundfield.generators.distributions import Distribution
from groundfield.generators.electrode_specs import (
    ElectrodeSpec,
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
)
from groundfield.world import World

__all__ = [
    "GroundingSystemSpec",
]


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    """Resolve a ``float | Distribution`` to ``float`` (sample once)."""
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


class GroundingSystemSpec(GeneratorConfig):
    """A complete grounding installation.

    Attributes
    ----------
    electrodes
        Ordered list of electrode specs. Order matters only for the
        choice of anchor (first *present* electrode). All present
        electrodes are bonded into one cluster.
    bond_conductor_type
        Conductor type used to bond the electrodes together. Default
        ``"bare_copper"`` matches the typical above-ground bonding
        strap inside a substation pit; switch to ``"pen"`` if you
        want the bond to be modelled like an insulated PEN cable.
    """

    electrodes: list[ElectrodeSpec] = Field(default_factory=list, min_length=0)
    bond_conductor_type: str = Field(
        default="bare_copper",
        description="Conductor type used to bond the electrodes together.",
    )

    def build_at(
        self,
        world: World,
        site_xy: tuple[float, float],
        name_prefix: str,
        rng: np.random.Generator,
    ) -> Optional[str]:
        """Materialise the grounding system inside ``world``.

        Parameters
        ----------
        world
            The :class:`World` to populate.
        site_xy
            Centre position ``(x, y)`` of this site (substation
            centre, KVS centre, building centre, …) in metres.
        name_prefix
            Prefix used for every created electrode name. The actual
            name is ``f"{name_prefix}_{kind}_{i}"``. Should be unique
            within the world.
        rng
            Random generator used to sample ``presence_prob`` and any
            geometric distributions.

        Returns
        -------
        str | None
            The name of the *anchor* electrode (the first present
            one). ``None`` if no electrode ended up being placed
            (every Bernoulli failed) — the caller may want to skip
            this site or raise.
        """
        cx, cy = site_xy
        created: list[str] = []
        for i, spec in enumerate(self.electrodes):
            # Bernoulli on presence
            p = _to_float(spec.presence_prob, rng)
            if p < 1.0 and rng.random() >= p:
                continue
            ex = cx + spec.offset_xy_m[0]
            ey = cy + spec.offset_xy_m[1]
            name = f"{name_prefix}_{spec.kind}_{i}"
            self._build_one(world, spec, ex, ey, name, rng)
            created.append(name)

        if not created:
            return None

        # Bond every non-anchor electrode to the anchor with a
        # bare-copper conductor. All bonds are short — the segment
        # length is set by the engine, not by us.
        anchor = created[0]
        for k, name in enumerate(created[1:], start=1):
            create_conductor(
                world,
                name=f"{name_prefix}_bond_{k}",
                start=anchor,
                end=name,
                conductor_type=self.bond_conductor_type,
            )
        return anchor

    # ------------------------------------------------------------------
    # Per-kind dispatcher
    # ------------------------------------------------------------------

    def _build_one(
        self,
        world: World,
        spec: ElectrodeSpec,
        ex: float,
        ey: float,
        name: str,
        rng: np.random.Generator,
    ) -> None:
        if isinstance(spec, RodElectrodeSpec):
            create_electrode(
                world, "rod", name=name,
                position=(ex, ey, _to_float(spec.depth_m, rng)),
                length=_to_float(spec.length_m, rng),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, RingElectrodeSpec):
            create_electrode(
                world, "ring", name=name,
                center=(ex, ey, _to_float(spec.depth_m, rng)),
                radius=_to_float(spec.radius_m, rng),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, StripElectrodeSpec):
            length = _to_float(spec.length_m, rng)
            angle = math.radians(_to_float(spec.orientation_deg, rng))
            depth = _to_float(spec.depth_m, rng)
            sx = ex - 0.5 * length * math.cos(angle)
            sy = ey - 0.5 * length * math.sin(angle)
            tx = ex + 0.5 * length * math.cos(angle)
            ty = ey + 0.5 * length * math.sin(angle)
            create_electrode(
                world, "strip", name=name,
                start=(sx, sy, depth),
                end=(tx, ty, depth),
                wire_radius=spec.wire_radius_m,
            )
            return
        if isinstance(spec, FoundationElectrodeSpec):
            depth = _to_float(spec.depth_m, rng)
            if spec.size_xy_m is not None:
                dx, dy = spec.size_xy_m
            else:
                size = _to_float(spec.size_m, rng)
                dx, dy = size, size
            # ``style="ring"`` forces a perimeter-only realisation —
            # GridMeshElectrode with n_x = n_y = 1 is exactly that
            # (one mesh cell = the four perimeter wires).
            if spec.style == "ring":
                nx, ny = 1, 1
            else:
                nx, ny = spec.n_x, spec.n_y
            create_electrode(
                world, "grid_mesh", name=name,
                corner=(ex - dx / 2.0, ey - dy / 2.0, depth),
                size=(dx, dy),
                n_x=nx,
                n_y=ny,
                wire_radius=spec.wire_radius_m,
            )
            return
        raise TypeError(
            f"GroundingSystemSpec.build_at: unknown electrode spec "
            f"{type(spec).__name__}."
        )
