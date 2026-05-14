"""Building-type specifications for generator-built worlds.

A :class:`BuildingTypeSpec` carries everything the generator needs
to know about *one class of building*:

* a ``name`` (used as the key in the generator-level ``counts``
  mapping and as a name prefix for the created electrodes),
* a :class:`GroundingSystemSpec` describing the building's
  earthing installation (foundation, rod, mesh, strip, ring, or any
  combination — every electrode optional via ``presence_prob``),
* an optional ``plot_size_m`` so future placement strategies can
  reserve space per type.

Typical predefined types (see the helper builders in this module):

* ``residential`` — single-family house with a foundation
  electrode.
* ``small_industry`` — small commercial / industrial building with
  foundation + driven rod.
* ``medium_industry`` — medium commercial with a larger foundation
  plus several rods.
* ``large_industry`` — large industrial site with a ring around the
  building plus a grid mesh and several rods. *Optional* fields are
  built in so an actual site can wire in a real geometry.

The default catalog returned by :func:`default_building_catalog` is
a sensible starting point; users tailor it to their study by
copying the resulting list and editing entries.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from groundfield.generators.base import GeneratorConfig
from groundfield.generators.electrode_specs import (
    FoundationElectrodeSpec,
    RingElectrodeSpec,
    RodElectrodeSpec,
    StripElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import GroundingSystemSpec

__all__ = [
    "BuildingTypeSpec",
    "default_building_catalog",
]


class BuildingTypeSpec(GeneratorConfig):
    """Definition of one building type.

    Attributes
    ----------
    name
        Identifier used as the lookup key in
        :attr:`groundfield.generators.tn_network.TnNetworkConfig.building_counts`
        and as a prefix for electrode names.
    grounding
        :class:`GroundingSystemSpec` describing the type's earthing
        installation. Every electrode in the system is optional via
        ``presence_prob`` and may carry distributions for its
        geometric parameters.
    plot_size_m
        Optional typical plot size $(dx, dy)$ in metres. v1 of the
        TN network generator does not consume this — it is exposed
        so future placement strategies (street-grid, OSM-driven)
        can pack buildings with type-aware footprints.
    description
        Free-form note describing the type (use case, source of
        the geometry, …). Stored verbatim, not validated.
    """

    name: str = Field(..., description="Building-type identifier.")
    grounding: GroundingSystemSpec = Field(
        default_factory=GroundingSystemSpec,
        description="Earthing system spec for this type.",
    )
    plot_size_m: Optional[tuple[float, float]] = Field(
        default=None,
        description="Optional plot size (dx, dy) in m for placement-aware sweeps.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Free-form documentation string.",
    )


# ---------------------------------------------------------------------
# Default catalog — starting point
# ---------------------------------------------------------------------


def default_building_catalog() -> list[BuildingTypeSpec]:
    """Return a sensible starting catalog of four building types.

    Returns
    -------
    list[BuildingTypeSpec]
        Four entries: ``residential``, ``small_industry``,
        ``medium_industry``, ``large_industry``. Geometries follow
        rough European-LV practice; tune via copy + edit before
        running serious studies.
    """
    return [
        BuildingTypeSpec(
            name="residential",
            description="Single-family house — foundation electrode only.",
            plot_size_m=(15.0, 15.0),
            grounding=GroundingSystemSpec(
                electrodes=[
                    FoundationElectrodeSpec(
                        size_m=10.0,
                        depth_m=0.8,
                        n_x=2, n_y=2,
                    ),
                ],
            ),
        ),
        BuildingTypeSpec(
            name="small_industry",
            description="Small commercial — foundation plus one driven rod.",
            plot_size_m=(20.0, 20.0),
            grounding=GroundingSystemSpec(
                electrodes=[
                    FoundationElectrodeSpec(
                        size_m=12.0, depth_m=0.8, n_x=2, n_y=2,
                    ),
                    RodElectrodeSpec(
                        length_m=2.0, depth_m=0.0,
                        offset_xy_m=(7.0, 0.0),
                    ),
                ],
            ),
        ),
        BuildingTypeSpec(
            name="medium_industry",
            description="Medium commercial — larger foundation plus four rods.",
            plot_size_m=(30.0, 30.0),
            grounding=GroundingSystemSpec(
                electrodes=[
                    FoundationElectrodeSpec(
                        size_m=20.0, depth_m=0.8, n_x=3, n_y=3,
                    ),
                    *rod_circle(n=4, radius_m=12.0, length_m=2.5),
                ],
            ),
        ),
        BuildingTypeSpec(
            name="large_industry",
            description=(
                "Large industrial site — perimeter ring, internal grid mesh, "
                "and eight driven rods spaced around the ring."
            ),
            plot_size_m=(60.0, 60.0),
            grounding=GroundingSystemSpec(
                electrodes=[
                    RingElectrodeSpec(radius_m=25.0, depth_m=0.8),
                    FoundationElectrodeSpec(
                        size_m=30.0, depth_m=0.8, n_x=4, n_y=4,
                    ),
                    *rod_circle(n=8, radius_m=25.0, length_m=3.0),
                    StripElectrodeSpec(
                        length_m=40.0, depth_m=0.8,
                        orientation_deg=0.0,
                        offset_xy_m=(0.0, 28.0),
                    ),
                    StripElectrodeSpec(
                        length_m=40.0, depth_m=0.8,
                        orientation_deg=0.0,
                        offset_xy_m=(0.0, -28.0),
                    ),
                ],
            ),
        ),
    ]
