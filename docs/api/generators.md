# Generators

The ``generators`` subpackage is the **factory layer** that converts
high-level parameter sets into a fully populated
:class:`groundfield.World`. It is the bridge between *parameter
studies* (typical, future Monte-Carlo runs) and the *physics layer*
(soil, electrodes, conductors, solver).

## Mathematical / physical context

For typical the relevant physics is

$$
\nabla \cdot \big( \sigma(z) \, \nabla \Phi \big) \;=\; 0
\quad \text{in } \mathbb{R}^3 \setminus \text{electrodes},
$$

with two-layer soil $\sigma(z)$ and Dirichlet conditions on the
metallic surfaces. Below 1 kHz the inductive correction is added
on top via Carson (ADR-0005) or Sommerfeld (ADR-0006). The role
of a generator is purely topological: it produces a concrete
electrode-and-conductor layout the solver can integrate over. No
physics happens in this subpackage.

The typical parameter axes that map directly onto
:class:`TnNetworkConfig` fields are summarised below. ``TnNetworkConfig`` has been
refactored onto a composable spec layer (ADR-0009 v2); building
mixes are now declared via ``building_types`` (a catalog of
:class:`BuildingTypeSpec`) and ``building_counts`` (per-type
quantities), and every grounding system is built from a list of
:class:`ElectrodeSpec` entries:

| Axis | default grid | Config field |
|------|----------|--------------|
| Single-family houses $n_\text{EFH}$ | 5, 10, 30, 80, 200 | ``building_counts["residential"]`` |
| Small commercial buildings | 0, 1, 5, 10 | ``building_counts["small_industry"]`` |
| Medium commercial buildings | 0, 1, 2, 5 | ``building_counts["medium_industry"]`` |
| Cable cabinets | placement spec | ``kvs.electrodes`` + ``placement`` |
| Substation grounding | ring / rod-circle / strip / foundation | ``substation.electrodes`` |
| Upper-layer resistivity $\rho_1$ | 30, 100, 200, 500, 1000 Ω·m | ``soil.rho_1`` |
| Lower-layer resistivity $\rho_2$ | 30, 100, 200 Ω·m | ``soil.rho_2`` |
| Layer thickness $h_1$ | 5, 10, 30 m | ``soil.h_1`` |

!!! info "Module rename"
    The module was originally called ``groundfield.generators.tn_ortsnetz``
    and exposed the classes ``TnOrtsnetzGenerator`` /
    ``TnOrtsnetzConfig``. The English-only naming convention renamed
    the module to ``groundfield.generators.tn_network`` with
    :class:`TnNetworkGenerator` / :class:`TnNetworkConfig`. The old
    module path still works as a deprecated re-export and emits a
    ``DeprecationWarning``. Prefer importing from the top-level
    ``groundfield`` namespace.

Every numeric field accepts either a fixed value or a
:class:`Distribution` (continuous: Uniform, Normal, LogNormal,
Weibull; discrete: Discrete, Categorical, plus the trivial
Constant). ``cfg.sample(rng)`` resolves every distribution
reproducibly given a fixed seed.

## Validity envelope

* Frequency: $f \le 1\,\mathrm{kHz}$ (quasi-static).
* Soil: linear, isotropic, layered. Saturation / ionisation are
  not modelled.
* Topology of :class:`TnNetworkGenerator`: radial-with-trunk
  via cable cabinets — every house connects to its nearest cable
  cabinet, every cable cabinet connects to the substation. Real
  street layouts are now supported via
  :class:`~groundfield.geo.placement.OsmBuildingPlacement` (see
  [ADR-0011](../adr/0011-osm-building-footprints.md) and the
  [geo API](geo.md)).
* Per-building-type grounding is now type-driven via
  :class:`BuildingTypeSpec` and supports multi-electrode systems
  (foundation + extra rod, ring + grid + strips, …) with
  per-electrode ``presence_prob`` for stochastic fleets.

## Foundation electrodes: orientation and concrete shell

Two extensions to :class:`FoundationElectrodeSpec` since 0.6.0 round
out the AP1 modelling envelope:

* ``orientation_deg: float | None = None`` rotates the foundation
  rectangle around its centre. ``None`` and ``0.0`` both keep the
  historic axis-aligned :class:`GridMeshElectrode` fast path; any
  other value synthesises the foundation from rotated
  :class:`StripElectrode`s and bonds them internally. Set
  automatically from the OMBR of an OSM polygon when the
  configured placement is
  :class:`~groundfield.geo.placement.OsmBuildingPlacement`.
* ``concrete_rho_ohm_m: float | AnyDistribution | None = None`` and
  ``concrete_thickness_m`` activate the **concrete-encasement model**
  ([ADR-0012](../adr/0012-foundation-concrete-encasement.md)).
  ``None`` keeps the wire-in-soil baseline; setting
  ``concrete_rho_ohm_m`` switches to the cylindrical Sunde-shell
  model with the chosen radial thickness. The
  ``concrete_model: Literal["lumped", "distributed"]`` discriminator
  selects between a lumped series resistance on the PEN service
  drop (V1, default — zero solver-side change, exact for the
  cluster impedance when current distributes uniformly) and a
  per-segment diagonal augmentation in the
  ``image`` / ``image_2layer`` backends (V2 — correct for non-uniform
  current distributions and for the surface potential right at
  the building wall). Stochastic moisture is supported via
  ``concrete_rho_ohm_m=Discrete(values=[50, 150, 500, 2000],
  weights=[0.25, 0.40, 0.25, 0.10])``.

The other electrode kinds (`Rod`, `Ring`, `Strip`, `Mesh`,
`GridMesh`) do **not** carry the concrete fields by design — they
correspond to electrodes that sit in trenches or driven holes,
never in a concrete strip foundation.

## Architecture

```
GeneratorConfig (Pydantic v2)
└── per-generator subclass (TnNetworkConfig, MvStrandConfig, ...)
    ├── numeric fields:   T | AnyDistribution
    ├── categorical:      str | Categorical
    └── nested configs (composable):
        ├── SoilSpec        (Homogeneous | TwoLayer | MultiLayer)
        ├── PlacementSpec   (Manhattan | Explicit)
        ├── GroundingSystemSpec (list[ElectrodeSpec])
        │   └── ElectrodeSpec (Rod | Ring | Strip | Foundation)
        └── BuildingTypeSpec (name + grounding + plot_size)

WorldGenerator[Generic[C]]   (ABC)
└── per-generator subclass (TnNetworkGenerator, ...)
    └── .build(cfg)  -> World
        strict mode:  guarded by ._assert_resolved(cfg)
        lazy mode:    resolves Distribution fields on demand
                      (TnNetworkGenerator chooses lazy mode)

WorldGenerator.sample_world(rng) -> tuple[World, ResolvedConfig]
```

Reproducibility: persist the *resolved* config alongside the
result. With the same seed and the same library version the
``World`` is bit-exactly reproducible.

Lazy vs. up-front resolution
----------------------------
:class:`TnNetworkGenerator` deliberately supports two patterns:

* ``gen.build(cfg)`` — lazy. Top-level numeric distributions get one
  draw per call. Per-instance distributions like a
  :class:`Categorical` ``house_electrode.kind`` get a fresh draw
  per house, producing a real mix.
* ``gen.sample_world(rng)`` — up-front. ``cfg.sample`` collapses
  every distribution to a single value, including the Categorical;
  *all* houses then share the resolved kind. Use this when you need
  a persistable resolved config for one Monte-Carlo realisation.

## Distribution catalogue

| Class | Backend | Typical use |
|-------|---------|-----------------|
| ``Constant`` | trivial | placeholder |
| ``Uniform(low, high)`` | numpy | bounded continuous parameters |
| ``Normal(mean, std, truncate_low?, truncate_high?)`` | rejection sampling | engineering tolerances |
| ``LogNormal(mu, sigma)`` (or ``LogNormal.from_moments``) | numpy | resistivities, sizes |
| ``Weibull(shape, scale)`` | numpy | wear / lifetime |
| ``Discrete(values, weights?)`` | numpy choice | ``n_efh ∈ {5,10,30,80,200}`` |
| ``Categorical(values, weights?)`` | numpy choice | electrode-kind mix per house |

All seven distributions are JSON-serialisable through a
discriminator field ``kind``. Persisting a stochastic config
preserves every distribution exactly; replaying with the same
seed yields the same world.

See [ADR-0009](../adr/0009-world-generators.md) for the full
design rationale, validation programme, and roadmap for
follow-up generators.

## Example

Two equivalent ways to drive :class:`TnNetworkGenerator`. Both
build a fully wired :class:`World` ready for ``world.solve(engine)``.

```python
import numpy as np
import groundfield as gf

# 1. Deterministic build — every parameter fixed.
cfg = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(rho_1=100.0, rho_2=500.0, h_1=2.0),
    building_counts={"residential": 10},  # 10 single-family houses
    source_magnitude_A=1.0,
)
gen = gf.TnNetworkGenerator()
world = gen.build(cfg)

# 2. Stochastic build — Categorical electrode kind per house,
#    Uniform soil resistivity, reproducible under a fixed seed.
cfg_stoch = gf.TnNetworkConfig(
    soil=gf.TwoLayerSoilSpec(
        rho_1=gf.Uniform(low=80.0, high=300.0),
        rho_2=500.0,
        h_1=2.0,
    ),
    building_counts={"residential": 30},
    source_magnitude_A=1.0,
)

rng = np.random.default_rng(seed=42)

# Lazy resolution — Categorical fields draw once per house,
# producing a real electrode-kind mix across the fleet.
world_lazy = gen.build(cfg_stoch, rng=rng)

# Strict / up-front resolution — every distribution collapses to a
# single value before build, so all houses share the same kind.
# Returns the resolved config too, which can be persisted alongside
# the result for bit-exact replay.
world_strict, resolved_cfg = gen.sample_world(rng=np.random.default_rng(seed=42))
```

Persist the resolved config (``cfg.model_dump_json()``) or the
seed alongside the simulation output to guarantee reproducibility.

## API reference

::: groundfield.generators
