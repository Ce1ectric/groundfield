"""OSM-driven placement strategy for :mod:`groundfield.generators`.

:class:`OsmBuildingPlacement` is a :class:`generators.placement.PlacementSpec`
variant: it returns building-centre positions in metres and additionally
exposes the corresponding :class:`BuildingFootprint` per site so the
generator can derive foundation-electrode dimensions from the polygon.

Two design constraints from ADR-0011 are enforced here:

* **No HTTP calls.** The class is constructed from an *already
  resolved* list of :class:`BuildingFootprint` instances. The
  network round-trip is the caller's responsibility (typically
  :func:`groundfield.geo.osm.query_and_project`). This keeps a
  generator config JSON-serialisable and replayable without
  internet access.
* **Deterministic ordering.** ``generate`` returns positions in
  *declared* order so that the *k*-th building type count maps to
  the *k*-th footprint. A future random-shuffle option can be
  added but must be opt-in.

The polygon access path used by the generator is :meth:`footprint_at`,
not direct attribute access on the union members — the generator
checks ``hasattr(placement, "footprint_at")`` once. New placement
variants can opt in to footprint-driven foundations the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from groundfield.geo.footprint import BuildingFootprint

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "OsmBuildingPlacement",
]


class OsmBuildingPlacement(BaseModel):
    """Placement driven by a list of pre-projected building footprints.

    The class implements the same ``generate(n, rng) -> list[(x, y)]``
    interface as :class:`generators.placement.ManhattanGridPlacement`
    and :class:`generators.placement.ExplicitPlacement`, so any
    consumer that accepts a :class:`PlacementSpec` will accept this
    placement too (once it has been added to the discriminated union;
    see Task 3 in ADR-0011's implementation plan).

    Parameters
    ----------
    footprints
        Pre-projected building polygons in the local ENU frame.
        Order is preserved; the *k*-th footprint corresponds to the
        *k*-th building site.
    min_area_m2
        Skip footprints whose area is strictly below this threshold
        on :meth:`generate`. The default of 16 m² is roughly a
        garden shed; everything smaller is almost certainly OSM
        noise rather than a residential structure with a
        foundation electrode.
    selection
        How to choose footprints when the generator requests fewer
        than are available:

        * ``"first_n"`` (default) — take the first ``n`` in
          declared order. Fully deterministic.
        * ``"all"`` — return all footprints regardless of ``n``.
          Useful when the user wants every visible building to
          appear in the world; ``n`` becomes a soft hint.

    Notes
    -----
    The class is intentionally a plain Pydantic ``BaseModel`` with
    a ``kind`` discriminator field; it does **not** inherit from a
    common ``PlacementSpec`` ABC because the existing union uses
    structural typing via the ``kind`` literal. Adding this class
    to ``generators.placement.PlacementSpec`` is a one-line union
    update tracked as a follow-up task.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["osm"] = "osm"
    footprints: list[BuildingFootprint] = Field(
        default_factory=list,
        description="Pre-projected building footprints (local ENU, metres).",
    )
    min_area_m2: float = Field(
        default=16.0,
        ge=0.0,
        description=(
            "Filter footprints below this area (m²) on ``generate``. "
            "Default matches a typical Gartenhaus."
        ),
    )
    selection: Literal["first_n", "all"] = Field(
        default="first_n",
        description="Subset rule when ``n`` < len(footprints).",
    )

    # -----------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------

    @field_validator("footprints")
    @classmethod
    def _normalise(
        cls, value: list[BuildingFootprint]
    ) -> list[BuildingFootprint]:
        # Defensive copy so a mutable caller list cannot mutate the
        # validated model in place. Pydantic v2 freezes by virtue of
        # ``BuildingFootprint`` being frozen, but the outer list is
        # not frozen — copy it once.
        return list(value)

    # -----------------------------------------------------------------
    # Filtered view
    # -----------------------------------------------------------------

    def _filtered(self) -> list[BuildingFootprint]:
        """Internal: footprints surviving the ``min_area_m2`` filter,
        in declared order."""
        if self.min_area_m2 <= 0.0:
            return list(self.footprints)
        return [
            fp for fp in self.footprints if fp.area_m2() >= self.min_area_m2
        ]

    # -----------------------------------------------------------------
    # PlacementSpec interface
    # -----------------------------------------------------------------

    def generate(
        self,
        n: int,
        rng: "np.random.Generator",  # unused — kept for interface parity
    ) -> list[tuple[float, float]]:
        """Return ``n`` site positions as polygon centroids.

        Parameters
        ----------
        n
            Number of sites the generator wants to populate.
        rng
            Unused. Accepted for interface parity with the other
            :class:`PlacementSpec` variants, all of which take an
            RNG so the generator can call them uniformly.

        Returns
        -------
        list[tuple[float, float]]
            Polygon centroids in metres. Length is ``min(n,
            len(filtered))`` when ``selection == "first_n"`` and
            equal to ``len(filtered)`` when ``selection == "all"``
            (``n`` is then ignored, with a soft contract).

        Raises
        ------
        ValueError
            If ``n`` exceeds the available footprints **and**
            ``selection == "first_n"``. The generator can decide
            whether to widen its search radius or lower its
            ``n``-target.
        """
        filtered = self._filtered()
        if self.selection == "all":
            return [fp.centroid_xy_m() for fp in filtered]
        if n > len(filtered):
            raise ValueError(
                f"OsmBuildingPlacement.generate: requested {n} sites "
                f"but only {len(filtered)} footprints are available "
                f"after the min_area_m2={self.min_area_m2} filter."
            )
        return [filtered[i].centroid_xy_m() for i in range(n)]

    # -----------------------------------------------------------------
    # Footprint-driven hook used by ``TnNetworkGenerator``
    # -----------------------------------------------------------------

    def footprint_at(self, i: int) -> Optional[BuildingFootprint]:
        """Return the footprint associated with site ``i``.

        Parameters
        ----------
        i
            Site index, matching the order returned by
            :meth:`generate`.

        Returns
        -------
        BuildingFootprint or None
            The footprint at index ``i``, or ``None`` when ``i`` is
            out of range. ``None`` rather than raising lets the
            consumer fall back to the spec-defined geometry without
            special-case logic.
        """
        filtered = self._filtered()
        if 0 <= i < len(filtered):
            return filtered[i]
        return None

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def __len__(self) -> int:
        """Number of footprints after applying ``min_area_m2``."""
        return len(self._filtered())
