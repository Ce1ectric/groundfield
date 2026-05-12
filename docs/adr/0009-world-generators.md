# ADR-0009: World generators and stochastic parameter distributions

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-05-08 |
| **Deciders** | Christian Ehlert |
| **Scope** | `groundfield.generators` (new subpackage); AP1 parameter
              studies; foundation for the `ParameterSweep` API |

## Context

`groundfield` now has the full physics stack (image / image_2layer /
image_nlayer / cim / mom / mom_sommerfeld / bem / fem,
inductive coupling, Carson, Sommerfeld, cross-layer, distributed
conductors) and a closed bridge to `groundinsight` via the `rho-f`
fit and the `BusType` exporter (ADR-0008). What is missing is the
**factory layer** that converts AP1-style parameters
(see *AP1_tn_ortsnetz.md* — number of single-family houses
$n_\text{EFH} \in \{5, 10, 30, 80, 200\}$, small/medium commercial
buildings, cable-cabinet quota, two-layer soil with
$\rho_1, \rho_2, h_1$) into a fully populated `World`. Without it
each parameter combination is hand-coded; with it the same
combination is one line and the parameter sweep across the AP1
table is a straightforward outer loop.

The user request additionally asks for two non-negotiable features:

1. **Multiple generators** are anticipated (TN-Ortsnetz first, but
   later e.g. an MV-strand generator, or an
   open-building-map-driven generator). The framework must be
   plug-extensible, not a single hard-coded function.
2. **Stochastic parameters**. Each numeric input may be fixed or
   sampled from a distribution (Normal, Weibull, LogNormal,
   Uniform, ...). Every sample must be reproducible (RNG seed) and
   the resolved configuration must be persistable (so a Monte
   Carlo run can be re-executed bit-exactly).

## Decision

### Architecture: ABC + per-generator Pydantic config

```
class WorldGenerator(ABC, Generic[C: GeneratorConfig]):
    cfg: C
    def __init__(self, cfg: C, *, rng: int | np.random.Generator | None = None): ...

    @abstractmethod
    def build(self, cfg: C | None = None) -> World: ...

    def sample_world(self, rng=None) -> tuple[World, C]: ...
```

Every concrete generator (`TnNetworkGenerator`, future
`MvStrandGenerator`, ...) ships its own `GeneratorConfig` subclass —
a Pydantic v2 model with full type information and defaults. The
generator class is thin; it takes a config, optionally samples any
stochastic fields via `cfg.sample(rng)`, and constructs the `World`.

Rationale (vs. a flat function `build_tn_ortsnetz(**params)`):

- Pydantic gives us validation, JSON dump/load, and IDE-grade
  autocompletion for free.
- Sweeps and CLI tools talk to the abstract `WorldGenerator`
  interface; no per-generator dispatch.
- Persisting a config is a one-liner; the resulting JSON is the
  *experimental record* of an AP1 run.

### Stochastic parameters: `Distribution` Pydantic classes

Every numerical field of a `GeneratorConfig` is typed as a discriminated
union `T | Distribution` where `Distribution` is the abstract base of
the seven first-class distributions:

| Class | Backend | Use case |
|-------|---------|----------|
| `Constant(value)` | trivial | placeholder for "act like a distribution" |
| `Uniform(low, high)` | `np.random.Generator.uniform` | bounded continuous |
| `Normal(mean, std, truncate_low?, truncate_high?)` | rejection sampling | engineering tolerances |
| `LogNormal(mu, sigma)` (or `mean_phys, std_phys`) | `rng.lognormal` | resistivities, sizes |
| `Weibull(shape, scale)` | `rng.weibull` × scale | wind/wear/lifetime |
| `Discrete(values, weights?)` | `rng.choice` | AP1 grid `n_efh ∈ {5,10,30,80,200}` |
| `Categorical(values, weights?)` | `rng.choice` (string) | electrode-kind mix |

Every distribution is a Pydantic v2 model with a `.sample(rng) -> Any`
method. Distributions are JSON-serialisable through a discriminator
field `kind: Literal["uniform", "normal", ...]`. A run can therefore be
persisted as

```json
{
  "n_efh": {"kind": "discrete", "values": [5, 10, 30, 80, 200]},
  "rho_1": {"kind": "lognormal", "mu": 5.0, "sigma": 0.7},
  "h_1":   {"kind": "uniform", "low": 2.0, "high": 10.0},
  "trafo_ring_radius_m": 4.0
}
```

and replayed bit-exactly given a seed.

### Resolution: Union field + `cfg.sample(rng)` traversal

`GeneratorConfig.sample(rng)` walks `self.__dict__` and replaces every
`Distribution` instance with its sampled value, recursing into nested
`GeneratorConfig` fields. The result is a *resolved* config — same
type, but every field is a constant.

`WorldGenerator` ships an `_assert_resolved(cfg)` helper that raises
`ValueError` when a config still carries `Distribution` fields. Whether
to call it inside `build` is a **per-generator choice**:

* **Strict mode** — `build` calls `_assert_resolved` and refuses any
  unresolved config. Stochastic configurations must go through
  `sample_world(rng)` (or a manual `cfg.sample(rng)`) first. Best
  for generators where every distribution is conceptually a
  Monte-Carlo axis (one draw per realisation).

* **Lazy mode** — `build` does not call `_assert_resolved` and
  resolves distributions opportunistically as values are needed.
  Top-level numeric distributions get one draw per `build` call;
  *per-instance* distributions (e.g. a `Categorical` electrode kind
  that the user wants drawn afresh per house) survive `cfg` and are
  sampled inside the relevant inner loop. `TnNetworkGenerator` uses
  this mode so that `Categorical(values=["foundation", "rod"])`
  in `house_electrode.kind` produces a real per-house mix when
  `gen.build(cfg)` is called directly.

Both modes coexist at the framework level. The user picks via the
generator they instantiate; `WorldGenerator.sample_world` works in
either case (it always pre-resolves through `cfg.sample`).

This keeps the type system simple: one config class per generator,
not two parallel "stochastic" and "deterministic" classes. The
trade-off is documented per generator: callers of `sample_world`
on a lazy-mode generator get *one* electrode kind for *all* houses
(because `cfg.sample` collapses the Categorical), while callers of
`build` directly get the per-house mix.

### Spec layer: composable building blocks

The first concrete generator (`TnNetworkGenerator`) sits on top of
five reusable spec layers. Each layer is independently
JSON-roundtrip-able and accepts distributions on every numerical
field; future generators (MV strand, OSM-driven distribution
network) reuse the same layers without code duplication.

**1. `electrode_specs` — single-electrode specifications.**
Discriminated union `ElectrodeSpec` with four members:
`RodElectrodeSpec`, `RingElectrodeSpec`, `StripElectrodeSpec`,
`FoundationElectrodeSpec`. Common fields: `presence_prob`
(Bernoulli per realisation), `offset_xy_m` (translation relative
to the site centre). Geometry parameters per kind are
`float | Distribution`. Helper `rod_circle(n, radius_m, …)` returns
N rods arranged on a circle — typical layout for the substation
*Tiefenerder*.

**2. `grounding.GroundingSystemSpec` — multi-electrode installation.**
Holds an ordered `electrodes: list[ElectrodeSpec]`. The method
`build_at(world, site_xy, name_prefix, rng)` materialises every
*present* electrode (Bernoulli on `presence_prob`), translates by
`offset_xy_m`, registers it with the world, and bonds all present
electrodes into one cluster via bare-copper conductors. Returns
the anchor (first present electrode) so the caller can wire PEN /
sources into the cluster.

This abstraction is shared by the substation, every cable cabinet,
and every building. A "ring + 4 rods + strip + foundation"
substation is one `GroundingSystemSpec` with seven
electrodes. A "70 % of houses have a foundation, 30 % also have
an additional rod" study sets `presence_prob=0.3` on the rod
spec inside the residential `BuildingTypeSpec`.

**3. `placement` — pluggable site placement on the 2-D plane.**
Discriminated union `PlacementSpec` with two members in v1:
`ManhattanGridPlacement` (regular street raster, optional
per-site jitter) and `ExplicitPlacement` (caller-supplied list of
$(x,y)$ tuples — useful for replaying a real map slice or for
deterministic small reference cases). Each implements
`generate(n, rng) -> list[(x, y)]`. Random scatter and cluster
placement are deferred to a follow-up.

**4. `soil_specs` — soil-model specifications.** Discriminated
union `SoilSpec` with three members: `HomogeneousSoilSpec`,
`TwoLayerSoilSpec`, `MultiLayerSoilSpec`. Each carries the same
fields as the underlying :class:`SoilModel` but allows
distributions; the method `to_soil(rng)` returns a fully numeric
:class:`SoilModel`. AP1's two-layer is the default.

**5. `building.BuildingTypeSpec` — building type definition.**
Bundles a `name`, a `GroundingSystemSpec`, and an optional
`plot_size_m`. Multiple types coexist in
`TnNetworkConfig.building_types`; the count per type is set via
`TnNetworkConfig.building_counts: dict[name, int | Distribution]`.
The default catalog returned by `default_building_catalog()`
ships AP1-typical entries for `residential`, `small_industry`,
`medium_industry`, and `large_industry`.

### Topology: medium detail (`TnNetworkGenerator` v2)

The generator composes the five spec layers into:

- A substation at `cfg.substation.position` with a
  `GroundingSystemSpec` of arbitrary composition.
- A configurable number of buildings (per type, per count) placed
  via `cfg.placement`. Each building runs its own
  `GroundingSystemSpec.build_at` — *per-instance* sampling of
  Bernoulli presence and geometric distributions, so a stochastic
  config naturally produces a heterogeneous fleet.
- $n_\text{KVS} = \lceil q \cdot n_\text{buildings} / 100 \rceil$
  cable cabinets (or a `fixed_count` override) placed via their
  own `cfg.kvs.placement`. Each KVS has its own
  `GroundingSystemSpec`.
- A **PEN backbone** as a *distributed conductor* (ADR-0003):
  substation → each KVS, each building → its nearest KVS
  (Manhattan metric). Configurable `coupling_to_soil`
  (`"isolated"` | `"galvanic"`) and `inductance_model` per
  ADR-0004.
- A current source attached to the substation cluster.

This level of detail captures every AP1 question (remote-injection
influence, inductive coupling between measurement leads and PEN/MV
shield, Carson relevance below 1 kHz, soil-layering monotonicity).
Higher fidelity (real street polygons, plot-level layouts,
foundation-shape diversity) is deferred to a follow-up generator
that will read from open building-map data.

## Validation programme

1. **Distributions** — each subclass passes:
   - reproducibility: same seed → identical sample sequence;
   - statistical sanity: 10 000 samples reproduce the analytical
     mean and variance to within 5 %;
   - JSON round-trip: `model_dump_json` + `model_validate_json`
     reproduces the distribution bit-exactly;
   - bound enforcement: `Normal` truncation never returns out-of-band
     values.
2. **Base framework** —
   - `cfg.sample(rng)` resolves all `Distribution` fields to constants
     and leaves non-distribution fields untouched;
   - `cfg.has_distributions()` is honest;
   - in *strict-mode* generators, `build(cfg_with_distributions)`
     raises a clear error via `_assert_resolved`;
   - in *lazy-mode* generators (e.g. `TnNetworkGenerator`),
     `build(cfg_with_distributions)` is supported and produces
     per-instance variation where designed;
   - JSON round-trip preserves the config.
3. **TnNetworkGenerator** —
   - the smallest preset (5 EFH) builds a `World` that solves
     successfully with `image_2layer`;
   - the largest preset (200 EFH) builds a `World` whose segment
     count stays below the configured budget (default 4 000);
   - parameter sweep over $n_\text{EFH} \times \rho_1$ produces
     monotonic cluster-impedance trends consistent with Dwight 1936
     (asymptotically $Z \propto \rho_1$ at low $f$);
   - the same seed produces a *bit-exact* identical world over
     repeated calls.

## Consequences

- The factory layer is now in place. The remaining roadmap items
  (`ParameterSweep` API, building-map ingestion) plug into the
  abstract `WorldGenerator` interface without further refactoring.
- `groundfield` gains a `scipy` dependency runtime use of
  `scipy.stats` (already a transitive dep through scipy itself; no
  pyproject change).
- The `Generic[C]` typing means tooling (mkdocstrings, mypy) keeps
  working for users of any generator.
- The discriminated-union JSON format is **versionless** for now;
  if the distribution catalogue changes incompatibly we will bump
  to a `schema_version` similar to ADR-0008.

## References

- AP1 work-package definition: `999_projektmanagement/arbeitspakete/AP1_tn_ortsnetz.md`.
- ADR-0003 — distributed conductors (PEN trunk leans on this).
- ADR-0008 — `BusType` export (downstream consumer of generator
  outputs after a `rho-f` fit).
- Vector fitting & rho-f standard form — what we eventually run
  on the generated networks.
