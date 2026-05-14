"""TN low-voltage distribution network generator (reference cases).

The generator composes the four reusable spec layers
(:mod:`~groundfield.generators.electrode_specs`,
:mod:`~groundfield.generators.grounding`,
:mod:`~groundfield.generators.placement`,
:mod:`~groundfield.generators.soil_specs`) plus the
per-:mod:`~groundfield.generators.building` type catalog into a
fully populated :class:`groundfield.World` representing a typical
TN low-voltage distribution network (substation, house connections,
cable cabinets).

Compared with the v1 prototype this generator now supports:

* **Heterogeneous substation grounding** — ring AND/OR rods AND/OR
  strip AND/OR foundation, in any combination, each with its own
  presence probability and stochastic geometry.
* **Heterogeneous KVS grounding** — same flexibility for cable
  cabinets; v1 forced a single rod.
* **Building-type catalog** — multiple :class:`BuildingTypeSpec`
  entries with independent grounding systems. The default catalog
  ships ``residential`` / ``small_industry`` / ``medium_industry``
  / ``large_industry`` and is fully customisable.
* **Pluggable placement** — Manhattan grid (with optional jitter) or
  explicit caller-supplied coordinates. Future strategies plug into
  the same :data:`PlacementSpec` discriminated union.
* **Pluggable soil model** — homogeneous, two-layer, or arbitrary
  multi-layer.

Lazy-resolution semantics
-------------------------
:class:`TnNetworkGenerator` keeps the v1 lazy-mode build:

* ``gen.build(cfg)`` resolves distributions opportunistically. Each
  *site* (substation, every KVS, every building) draws its
  ``presence_prob`` Bernoulli and samples its geometric
  distributions independently. This is what produces a real
  per-site mix when the user supplies population-level
  distributions (e.g. ``presence_prob=0.7`` for an "additional
  rod" electrode that 70 % of houses receive).
* ``gen.sample_world(rng)`` calls ``cfg.sample(rng)`` first, which
  collapses every distribution to one constant — including
  ``presence_prob``, which then becomes a deterministic
  flag. Useful for persistable Monte-Carlo realisations.

Geometric layout
----------------
* Substation at ``substation.position``. Its grounding system is
  built via :meth:`GroundingSystemSpec.build_at`.
* Buildings are placed via the configured :data:`PlacementSpec`.
  When ``placement`` is :class:`ManhattanGridPlacement`, buildings
  are generated in the order ``[type_0]*n_0 + [type_1]*n_1 + …`` —
  i.e. all buildings of the first type fill the grid first. This
  keeps the layout deterministic and reproducible. Use
  :class:`ExplicitPlacement` if you need a specific spatial mix.
* Cable cabinets are placed via their own
  :class:`PlacementSpec` (defaults to a Manhattan grid along the
  substation row). Their grounding follows :attr:`KvsConfig.grounding`.
* PEN backbone: each cable cabinet is connected to the substation;
  each building is connected to the *nearest* cable cabinet by
  Manhattan distance.

Validity envelope
-----------------
* Frequency: $f \\le 1\\,\\mathrm{kHz}$ (quasi-static).
* Soil: any of the spec'd models. default is two-layer.
* Topology: radial-with-trunk via cable cabinets — the
  open-building-map / real-street layout is roadmap.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Optional, Union

import numpy as np
from pydantic import Field

from groundfield.api import create_conductor, create_source, create_world
from groundfield.generators.base import (
    GeneratorConfig,
    WorldGenerator,
)
from groundfield.generators.building import (
    BuildingTypeSpec,
    default_building_catalog,
)
from groundfield.generators.distributions import AnyDistribution, Distribution
from groundfield.generators.electrode_specs import (
    RingElectrodeSpec,
    RodElectrodeSpec,
    rod_circle,
)
from groundfield.generators.grounding import GroundingSystemSpec
from groundfield.generators.measurement import (
    MeasurementLeadConfig,
    MeasurementSetupConfig,
)
from groundfield.generators.placement import (
    ExplicitPlacement,
    ManhattanGridPlacement,
    PlacementSpec,
)
from groundfield.generators.soil_specs import (
    HomogeneousSoilSpec,
    MultiLayerSoilSpec,
    SoilSpec,
    TwoLayerSoilSpec,
    materialise_soil,
)
from groundfield.world import World

__all__ = [
    "PenConfig",
    "SubstationConfig",
    "KvsConfig",
    "TnNetworkConfig",
    "TnNetworkGenerator",
]


# ---------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------


def _to_float(value: Union[float, Distribution], rng: np.random.Generator) -> float:
    if isinstance(value, Distribution):
        return float(value.sample(rng))
    return float(value)


def _to_int(value: Union[int, Distribution], rng: np.random.Generator) -> int:
    if isinstance(value, Distribution):
        return int(round(float(value.sample(rng))))
    return int(round(float(value)))


# ---------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------


def _default_substation_grounding() -> GroundingSystemSpec:
    """default substation grounding: ring + 4 rods on a 2 m circle."""
    return GroundingSystemSpec(
        electrodes=[
            RingElectrodeSpec(radius_m=4.0, depth_m=0.6),
            *rod_circle(n=4, radius_m=2.0, length_m=2.5),
        ],
    )


def _default_kvs_grounding() -> GroundingSystemSpec:
    """default KVS grounding: a single 1.5 m driven rod."""
    return GroundingSystemSpec(
        electrodes=[
            RodElectrodeSpec(length_m=1.5, depth_m=0.0),
        ],
    )


def _default_building_placement() -> PlacementSpec:
    """default building placement: Manhattan grid 25 × 30 m, 10 per row."""
    return ManhattanGridPlacement(
        spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=10,
        centre_xy=(0.0, 0.0),
    )


def _default_kvs_placement() -> PlacementSpec:
    """Default KVS placement: a wide Manhattan strip along the substation row."""
    return ManhattanGridPlacement(
        spacing_x_m=40.0, spacing_y_m=1.0, n_per_row=10,
        centre_xy=(0.0, 0.0),
    )


class SubstationConfig(GeneratorConfig):
    """Substation block — position + grounding system."""

    position: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Substation centre (x, y) in m.",
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_substation_grounding,
        description="Substation grounding system.",
    )


class KvsConfig(GeneratorConfig):
    """Cable cabinet (KVS) block — placement, count, grounding."""

    placement: PlacementSpec = Field(
        default_factory=_default_kvs_placement,
        description="Where to place the cable cabinets.",
    )
    quote_per_100_buildings: Union[float, AnyDistribution] = Field(
        default=5.0,
        description=(
            "Number of cable cabinets per 100 buildings. The actual KVS "
            "count is ceil(quote * n_buildings / 100), with a floor of 1."
        ),
    )
    fixed_count: Optional[Union[int, AnyDistribution]] = Field(
        default=None,
        description=(
            "Optional: pin the number of KVS to this value, ignoring "
            "``quote_per_100_buildings``. Useful for explicit-placement runs."
        ),
    )
    grounding: GroundingSystemSpec = Field(
        default_factory=_default_kvs_grounding,
        description="Per-KVS grounding system.",
    )


class PenConfig(GeneratorConfig):
    """PEN cable backbone (distributed conductor, ADR-0003)."""

    wire_radius_m: float = Field(default=0.005, description="PEN wire radius in m.")
    depth_m: float = Field(default=0.6, description="PEN burial depth in m.")
    coupling_to_soil: str = Field(
        default="isolated",
        description=(
            "'isolated' for typical insulated cable, 'galvanic' for bare "
            "copper / exposed shield."
        ),
    )
    segment_length_m: Optional[float] = Field(
        default=5.0,
        description=(
            "Discretisation segment length in m for the distributed "
            "conductor model. None keeps the conductor lumped."
        ),
    )
    inductance_model: Optional[str] = Field(
        default=None,
        description="ADR-0004 inductance model: 'neumann' or None.",
    )


# ---------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------


def _default_building_catalog() -> list[BuildingTypeSpec]:
    return default_building_catalog()


def _default_building_counts() -> dict[str, Union[int, AnyDistribution]]:
    """30 residential single-family houses, no commercial."""
    return {"residential": 30}


def _default_soil() -> SoilSpec:
    return TwoLayerSoilSpec()


class TnNetworkConfig(GeneratorConfig):
    """Top-level configuration for a TN low-voltage distribution network.

    Attributes
    ----------
    name
        World name.
    soil
        Soil-model spec — homogeneous, two-layer, or multi-layer
        (see :data:`~groundfield.generators.soil_specs.SoilSpec`).
    substation
        Substation block (position + grounding system).
    kvs
        Cable-cabinet block (placement, count, grounding).
    placement
        Building placement strategy. Default: Manhattan grid.
    building_types
        Catalog of building types available in this run. Each type
        carries its own grounding system. Defaults to a four-type
        catalog (residential, small/medium/large industry).
    building_counts
        Number of buildings per type. Keys must match a
        ``building_types[*].name``. Values may be fixed or
        :class:`Distribution`. Buildings are placed in catalog order.
    pen
        PEN backbone configuration.
    source_magnitude_A
        Driving current at the substation cluster, in A.
    """

    name: str = Field(default="tn_network", description="World name.")

    soil: SoilSpec = Field(
        default_factory=_default_soil,
        description="Soil model spec.",
    )

    substation: SubstationConfig = Field(
        default_factory=SubstationConfig,
        description="Substation block.",
    )

    kvs: KvsConfig = Field(
        default_factory=KvsConfig,
        description="Cable cabinet block.",
    )

    placement: PlacementSpec = Field(
        default_factory=_default_building_placement,
        description="Building placement strategy.",
    )

    building_types: list[BuildingTypeSpec] = Field(
        default_factory=_default_building_catalog,
        description="Catalog of building types available in this run.",
    )

    building_counts: dict[str, Union[int, AnyDistribution]] = Field(
        default_factory=_default_building_counts,
        description=(
            "Number of buildings per type. Keys must match "
            "building_types[*].name."
        ),
    )

    pen: PenConfig = Field(
        default_factory=PenConfig,
        description="PEN backbone block.",
    )

    source_magnitude_A: Union[float, AnyDistribution] = Field(
        default=1.0,
        description=(
            "Driving current at the substation cluster, in A (RMS). "
            "Interpretation depends on the study: for a fall-of-"
            "potential measurement this is the *test current* fed in "
            "by the measurement device (typically 1 A … 25 A); for a "
            "fault simulation this is the *single-phase ground-fault "
            "current* (typically a few hundred A to several kA). "
            "The solver is linear, so for normalised studies setting "
            "this to 1.0 lets every potential / cluster impedance be "
            "read directly as Ω. Accepts a :class:`Distribution` for "
            "stochastic sweeps (e.g. ``Discrete(values=[1.0, 5000.0])`` "
            "to compare measurement vs. fault on the same world)."
        ),
    )

    measurement: Optional[MeasurementSetupConfig] = Field(
        default=None,
        description=(
            "Optional grounding-resistance measurement setup "
            "(the galvanic fall-of-potential analysis + 2). When set, the generator adds "
            "the auxiliary current electrode, the voltage probe, "
            "and (optionally) the metallic feed / probe leads, and "
            "re-routes the source's return path through the "
            "auxiliary electrode. ``None`` keeps the historic "
            "behaviour: source returns to remote earth, no probe "
            "or aux electrode."
        ),
    )

    source_return_to: Optional[str] = Field(
        default=None,
        description=(
            "Optional **explicit** override for the source's "
            "``return_to`` electrode name. ``None`` (default) "
            "preserves the historic behaviour: ``return_to`` is "
            "derived automatically from :attr:`measurement` (the "
            "auxiliary current electrode) or left as ``None`` for "
            "remote-earth return. Setting this field takes "
            "precedence over the measurement-setup auto-routing — "
            ":meth:`TnNetworkGenerator.build` emits a "
            ":class:`UserWarning` when both are set so the "
            "precedence is explicit rather than silent (fourth "
            "2026-05-12 audit pass)."
        ),
    )

    source_kind: Literal["current", "voltage"] = Field(
        default="current",
        description=(
            "Source type at the substation. Only ``\"current\"`` "
            "(default) and ``\"voltage\"`` are accepted — typos like "
            "``\"voltage_\"`` are rejected at validation time with a "
            "Pydantic ``ValidationError`` rather than silently falling "
            "through to the default ``CurrentSource`` factory (fifth "
            "2026-05-13 audit pass). ``\"current\"`` is the right "
            "choice for the typical fall-of-potential measurement and "
            "for fault simulations; ``\"voltage\"`` is reserved for "
            "the rare multi-port tests."
        ),
    )


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------


class TnNetworkGenerator(WorldGenerator[TnNetworkConfig]):
    """Generator for TN low-voltage distribution network reference worlds.

    See the module docstring for the full pipeline. In one
    sentence: build soil → build substation grounding → place
    buildings (one per type-and-count) → build per-building
    grounding → place cable cabinets → build per-KVS grounding
    → wire the PEN backbone → attach the source.
    """

    def build(self, cfg: Optional[TnNetworkConfig] = None) -> World:
        cfg = cfg or self.cfg
        rng = self._rng

        # --- Soil + world ---
        soil = materialise_soil(cfg.soil, rng)
        world = create_world(name=cfg.name, soil=soil)

        # --- Substation grounding ---
        substation_anchor = cfg.substation.grounding.build_at(
            world,
            site_xy=cfg.substation.position,
            name_prefix="trafo",
            rng=rng,
        )
        if substation_anchor is None:
            raise ValueError(
                "TnNetworkGenerator: substation grounding produced zero "
                "electrodes. At least one electrode must have presence_prob > 0."
            )

        # --- Buildings ---
        building_assignments = self._resolve_building_counts(cfg, rng)
        n_buildings = sum(count for _, count in building_assignments)
        if n_buildings < 1:
            raise ValueError(
                "TnNetworkGenerator: total building count is 0. Populate "
                "building_counts with at least one type."
            )

        positions = cfg.placement.generate(n_buildings, rng)
        if len(positions) != n_buildings:
            raise RuntimeError(
                f"placement.generate returned {len(positions)} positions, "
                f"expected {n_buildings}."
            )

        building_anchors: list[tuple[str, tuple[float, float]]] = []
        idx = 0
        for type_idx, (btype, count) in enumerate(building_assignments):
            for k in range(count):
                site_xy = positions[idx]
                prefix = f"{btype.name}_{k}"
                anchor = btype.grounding.build_at(
                    world, site_xy=site_xy, name_prefix=prefix, rng=rng,
                )
                if anchor is not None:
                    building_anchors.append((anchor, site_xy))
                idx += 1

        if not building_anchors:
            raise ValueError(
                "TnNetworkGenerator: every building's grounding had zero "
                "present electrodes. Increase presence_prob."
            )

        # --- Cable cabinets ---
        n_kvs = self._resolve_kvs_count(cfg, n_buildings, rng)
        kvs_positions = cfg.kvs.placement.generate(n_kvs, rng)
        kvs_anchors: list[tuple[str, tuple[float, float]]] = []
        for k, site_xy in enumerate(kvs_positions):
            anchor = cfg.kvs.grounding.build_at(
                world, site_xy=site_xy, name_prefix=f"kvs_{k}", rng=rng,
            )
            if anchor is not None:
                kvs_anchors.append((anchor, site_xy))

        if not kvs_anchors:
            raise ValueError(
                "TnNetworkGenerator: every KVS grounding had zero present "
                "electrodes. Increase presence_prob or set fixed_count > 0."
            )

        # --- PEN backbone ---
        self._build_pen_backbone(
            world, cfg, substation_anchor, kvs_anchors, building_anchors,
        )

        # --- Optional measurement setup (the galvanic fall-of-potential analysis + 2) ---
        # Materialise the auxiliary current electrode, the voltage
        # probe, and (optionally) the metallic feed / probe leads.
        # The resulting return-path anchor (or ``None`` if no
        # measurement setup is configured) is forwarded to the
        # source so the test current physically returns through the
        # aux electrode.
        return_anchor = self._build_measurement_setup(
            world, cfg, substation_anchor,
        )

        # --- Source ---
        # ``source_magnitude_A`` may be a Distribution; resolve lazily
        # like every other top-level numeric field.
        #
        # ``cfg.source_return_to`` is the explicit user-side override
        # for the source's ``return_to``. If it is set together with a
        # measurement setup, warn that the auto-routed aux anchor is
        # being overridden — that override was silent before the
        # fourth 2026-05-12 audit pass.
        effective_return_to = return_anchor
        if cfg.source_return_to is not None:
            if (
                return_anchor is not None
                and cfg.source_return_to != return_anchor
            ):
                warnings.warn(
                    "TnNetworkConfig.source_return_to="
                    f"{cfg.source_return_to!r} takes precedence over "
                    "the measurement-setup auxiliary electrode "
                    f"({return_anchor!r}). The aux electrode and any "
                    "metallic feed/probe leads remain in the world, "
                    "but the source's return current is routed to "
                    "the user-supplied electrode instead.",
                    UserWarning,
                    stacklevel=2,
                )
            effective_return_to = cfg.source_return_to

        create_source(
            world, attached_to=substation_anchor,
            return_to=effective_return_to,
            kind=cfg.source_kind,
            magnitude=_to_float(cfg.source_magnitude_A, rng),
        )

        return world

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_building_counts(
        self, cfg: TnNetworkConfig, rng: np.random.Generator,
    ) -> list[tuple[BuildingTypeSpec, int]]:
        """Resolve the (type → count) mapping into an ordered list.

        Returns
        -------
        list of (BuildingTypeSpec, int)
            One entry per building type present in
            ``cfg.building_counts``. Order matches the catalog order
            in ``cfg.building_types``. Types missing from the
            catalog raise :class:`KeyError`.
        """
        catalog_by_name = {t.name: t for t in cfg.building_types}
        out: list[tuple[BuildingTypeSpec, int]] = []
        for name, count_spec in cfg.building_counts.items():
            if name not in catalog_by_name:
                raise KeyError(
                    f"building_counts['{name}'] has no matching entry in "
                    f"building_types (available: {list(catalog_by_name)})."
                )
            count = _to_int(count_spec, rng)
            if count > 0:
                out.append((catalog_by_name[name], count))
        # Order by catalog (deterministic) then by counts dict insertion
        # — `out` already follows insertion order of building_counts;
        # re-sort to catalog order so layout is stable across reruns
        # that change dict iteration.
        catalog_order = {t.name: i for i, t in enumerate(cfg.building_types)}
        out.sort(key=lambda pair: catalog_order[pair[0].name])
        return out

    def _resolve_kvs_count(
        self,
        cfg: TnNetworkConfig,
        n_buildings: int,
        rng: np.random.Generator,
    ) -> int:
        """Resolve the actual cable-cabinet count.

        Uses ``cfg.kvs.fixed_count`` if provided; otherwise computes
        ``ceil(quote * n_buildings / 100)`` with a floor of 1.
        """
        if cfg.kvs.fixed_count is not None:
            return max(1, _to_int(cfg.kvs.fixed_count, rng))
        quote = _to_float(cfg.kvs.quote_per_100_buildings, rng)
        return max(1, int(math.ceil(quote * n_buildings / 100.0)))

    def _build_measurement_setup(
        self,
        world: World,
        cfg: TnNetworkConfig,
        substation_anchor: str,
    ) -> Optional[str]:
        """Materialise the optional measurement setup.

        If ``cfg.measurement`` is ``None``, this is a no-op and the
        method returns ``None`` (which the caller forwards to
        :func:`create_source` as ``return_to``, leaving the source
        to return through remote earth).

        Otherwise this method:

        1. builds the auxiliary current electrode
           (``cfg.measurement.injection.grounding.build_at(...)``);
        2. builds the voltage probe
           (``cfg.measurement.probe.grounding.build_at(...)``);
        3. if configured, adds the metallic feed lead from the
           substation anchor to the aux anchor;
        4. if configured, adds the metallic probe lead from the
           substation anchor to the probe anchor;
        5. returns the aux anchor name, so the source's
           ``return_to`` is wired to the auxiliary electrode.
        """
        meas = cfg.measurement
        if meas is None:
            return None
        rng = self._rng

        # 1) Auxiliary current electrode (Hilfserder)
        aux_anchor = meas.injection.grounding.build_at(
            world,
            site_xy=meas.injection.position_xy,
            name_prefix="aux",
            rng=rng,
        )
        if aux_anchor is None:
            raise ValueError(
                "TnNetworkGenerator: measurement.injection.grounding "
                "produced zero electrodes. Increase presence_prob."
            )

        # 2) Voltage probe (Spannungssonde)
        probe_anchor = meas.probe.grounding.build_at(
            world,
            site_xy=meas.probe.position_xy,
            name_prefix="probe",
            rng=rng,
        )
        if probe_anchor is None:
            raise ValueError(
                "TnNetworkGenerator: measurement.probe.grounding "
                "produced zero electrodes. Increase presence_prob."
            )

        # 3) Metallic feed lead (Stromeinspeiseleitung), optional.
        if meas.injection.feed_lead is not None:
            self._build_measurement_lead(
                world,
                lead=meas.injection.feed_lead,
                start_anchor=substation_anchor,
                end_anchor=aux_anchor,
                name="meas_feed_lead",
            )

        # 4) Metallic probe lead (Spannungs-Messleitung), optional.
        if meas.probe.lead is not None:
            self._build_measurement_lead(
                world,
                lead=meas.probe.lead,
                start_anchor=substation_anchor,
                end_anchor=probe_anchor,
                name="meas_probe_lead",
            )

        return aux_anchor

    def _build_measurement_lead(
        self,
        world: World,
        *,
        lead: MeasurementLeadConfig,
        start_anchor: str,
        end_anchor: str,
        name: str,
    ) -> None:
        """Build one measurement lead as a finite-impedance ``Conductor``.

        The wire connects two existing anchor electrodes; ``depth_m``
        is the lead's $z$-coordinate at both endpoints (the engine
        takes the connection points of the anchors as the actual
        endpoints, so a lead from a deep ring electrode to a
        surface-rod electrode has a slight diagonal — see the
        module docstring of
        :mod:`groundfield.generators.measurement` for the validity
        envelope).
        """
        rng = self._rng
        kwargs: dict = dict(
            conductor_type=lead.conductor_type,
            wire_radius=lead.wire_radius_m,
            cross_section="from_radius",  # finite-impedance branch
            coupling_to_soil=lead.coupling_to_soil,
        )
        if lead.segment_length_m is not None:
            kwargs["discretize_segment_length"] = lead.segment_length_m
        if lead.inductance_model is not None:
            kwargs["inductance_model"] = lead.inductance_model
        # ``depth_m`` is currently informational — the lead's actual
        # endpoint depths come from the anchor electrodes' connection
        # points. For a future, fully depth-controlled routing the
        # builder would interpose surface-clamp electrodes; for v1
        # we keep the simpler model and document the limitation.
        _ = _to_float(lead.depth_m, rng)
        create_conductor(
            world,
            name=name,
            start=start_anchor,
            end=end_anchor,
            **kwargs,
        )

    def _build_pen_backbone(
        self,
        world: World,
        cfg: TnNetworkConfig,
        substation_anchor: str,
        kvs_anchors: list[tuple[str, tuple[float, float]]],
        building_anchors: list[tuple[str, tuple[float, float]]],
    ) -> None:
        """Build the PEN backbone: substation → KVS → buildings."""
        pen = cfg.pen
        common_kwargs: dict = dict(
            conductor_type="pen",
            wire_radius=pen.wire_radius_m,
            cross_section="from_radius",
            coupling_to_soil=pen.coupling_to_soil,
        )
        if pen.segment_length_m is not None:
            common_kwargs["discretize_segment_length"] = pen.segment_length_m
        if pen.inductance_model is not None:
            common_kwargs["inductance_model"] = pen.inductance_model

        # Substation → each KVS
        for kvs_name, _ in kvs_anchors:
            create_conductor(
                world, name=f"pen_main_{kvs_name}",
                start=substation_anchor, end=kvs_name,
                **common_kwargs,
            )
        # Each building → its nearest KVS (Manhattan metric)
        for building_name, (bx, by) in building_anchors:
            kvs_name, _ = min(
                kvs_anchors,
                key=lambda k: abs(k[1][0] - bx) + abs(k[1][1] - by),
            )
            create_conductor(
                world, name=f"pen_service_{building_name}",
                start=kvs_name, end=building_name,
                **common_kwargs,
            )
