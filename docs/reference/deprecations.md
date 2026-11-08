# Deprecations

**Policy** (since the calendar release 2026.11): a deprecated name,
parameter or behaviour keeps working and emits a `DeprecationWarning`
that names its replacement and the earliest release in which it may be
removed. It is removed no earlier than two releases **and** six months
after the release that announced it. Everything below was announced in
**2026.11.0** and may be removed in **2027.5** at the earliest
(`groundfield.utils.REMOVAL_VERSION`). Files written by 0.15.0 (world,
engine, result and generator-configuration JSON) still load; old JSON
keys are mapped with a warning.

Find deprecated names in your code by running it with
`python -W error::DeprecationWarning` (or `pytest -W
error::DeprecationWarning`); type checkers already report the old soil
keywords.

## Soil models

| Deprecated | Replacement | Accepted as |
|---|---|---|
| `HomogeneousSoil(resistivity=...)`, `.resistivity` | `rho` | keyword, JSON key, attribute, `model_copy(update=...)` key |
| `SoilLayer(resistivity=...)`, `.resistivity` | `rho` | same |
| `SoilLayer(thickness=...)`, `.thickness` | `h` (`None` for the bottom half-space) | same |

Passing an old and its new name together raises `ValueError`.

## Engine and solve

| Deprecated | Replacement | Note |
|---|---|---|
| `Engine.tolerance`, `Engine.max_iterations` | — | no backend reads them; a non-default value or attribute access warns |
| `Engine.sommerfeld_lambda_max_factor`, `sommerfeld_epsabs`, `sommerfeld_epsrel`, `sommerfeld_max_osc_panels` | `Engine(options={"mom_sommerfeld": {...}})` or `MomSommerfeldOptions` | accepted in the constructor, in JSON (a value equal to the default loads silently), as attribute and in `model_copy` |
| `World.solve(engine, snapshot_sources=False)` | `World.solve(engine)` | every entry point takes the source snapshot now; the opt-out saves nothing |
| `World.concrete_shell_corrections` (lumped shell registry, ADR-0012 V1) | the per-length coefficient `concrete_shell_coefficient_ohm_m` of the encased electrodes (`shell_coefficient_from_resistance` converts a lumped resistance) | the solver never read the registry; solving a world with entries warns; the field still loads from JSON |

## Backend names

| Deprecated | Runs | Note |
|---|---|---|
| `backend="cim"`, `solver.cim.solve_cim` | the `image` auto-dispatch (`image`, `image_2layer`) | bit-identical on every world compared; result metadata are those of the backend that ran (`cim_fit_used`, `cim_n_images`, `cim_rms`, `reduces_to` are gone) |
| `backend="bem"`, `solver.bem.solve_bem` | `mom` | it assembled the identical point-matching system (results move by at most 3.5e-16) |
| `backend="image_nlayer"` | the name of the `image` auto-dispatch for `MultiLayerSoil` | `Engine(backend="image")` stays silent; three or more distinct layers raise and name `mom_sommerfeld` |

The modules `groundfield.solver.cim` (with `fit_complex_images`) and
`groundfield.solver.bem` stay importable.

## Generators (English names, 10.69)

| Deprecated (v0.15.0) | Replacement |
|---|---|
| module `groundfield.generators.ortsnetz_builder` | `groundfield.generators.layout` |
| `OrtsnetzLayout` | `LvNetworkLayout` |
| `KvsPlacement` | `CabinetPlacement` |
| `KvsConfig` | `CabinetConfig` |
| `StarKvsTopology`, `kind="star_kvs"` | `StarCabinetTopology`, `kind="star_cabinet"` |
| `TnNetworkConfig.kvs` | `TnNetworkConfig.cabinets` |
| `TnNetworkConfig.source_magnitude_A` | `TnNetworkConfig.source_magnitude_a` |
| `CabinetConfig.quote_per_100_buildings` | `CabinetConfig.count_per_100_buildings` |
| `RadialTrunkTopology.slots_per_kvs` | `RadialTrunkTopology.slots_per_cabinet` |
| `RadialTrunkTopology.kvs_spacing_m` | `RadialTrunkTopology.cabinet_spacing_m` |
| `RadialTrunkTopology.plan_feeder_kvs_positions()` | `RadialTrunkTopology.plan_feeder_cabinet_positions()` |
| `LvNetworkLayout.add_kvs()` | `LvNetworkLayout.add_cabinet()` |
| `LvNetworkLayout.kvs_placements` | `LvNetworkLayout.cabinet_placements` |
| `from_osm(kvs_lat_lons=, kvs_xys=)` | `from_osm(cabinet_lat_lons=, cabinet_xys=)` |
| `to_world(kvs_grounding=)` | `to_world(cabinet_grounding=)` (also on `plan`) |
| `source_magnitude_A=` of `to_world`, `measured_grounding_impedance`, `verify_current_balance` | `source_magnitude_a=` |
| result keys `positive_side_A`, `auxiliary_side_A`, `expected_*_A`, `abs_error_*_A`, `tolerance_A` of `verify_current_balance` | `positive_side_a`, `auxiliary_side_a`, `expected_*_a`, `abs_error_*_a`, `tolerance_a` |
| `HomogeneousSoilSpec.resistivity` | `HomogeneousSoilSpec.rho` |
| `SoilLayerSpec.resistivity`, `SoilLayerSpec.thickness_m` | `SoilLayerSpec.rho`, `SoilLayerSpec.h` |

The older module alias `groundfield.generators.tn_ortsnetz`
(`TnOrtsnetzGenerator`, `TnOrtsnetzConfig`) keeps working with a
warning as before.

## Removed in 2026.11.0

Not deprecated but removed, because they were tools rather than
library API: the console scripts `release` and
`generate-third-party-licenses` and the installed top-level package
`scripts` (finding 10.02; run `python scripts/release.py` from a
checkout), and the commands `major|minor|patch` of the release tool
(replaced by `calver` and `set`).

## Checking the list

The deprecated names of this page are exercised by the test suite:
`tests/test_soil_models.py`, `tests/test_engine_options.py`,
`tests/test_solve_entry_points.py`, `tests/test_concrete_shell_registry.py`,
`tests/test_engine_aliases.py` and `tests/test_generator_naming.py`. A
quick check of one entry per group:

```python
import warnings

import groundfield as gf

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    soil = gf.HomogeneousSoil(resistivity=100.0)                 # soil keyword
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=3.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    result = gf.Engine(backend="bem", frequencies=[50.0]).solve(world)   # backend alias
    config = gf.TnNetworkConfig(source_magnitude_A=1.0)        # generator field

messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
assert len(messages) == 3 and all("2027.5" in m for m in messages)
assert soil.rho == 100.0 and config.source_magnitude_a == 1.0
assert result.metadata["effective_backend"] == "mom"
```
