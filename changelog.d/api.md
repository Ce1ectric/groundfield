### Added

- **Typed factories (10.68):** `create_electrode` has one `@overload` per
  electrode kind and returns the concrete class
  (`create_electrode(world, "ring", ...)` is a `RingElectrode`), so IDEs
  show the parameters of the chosen kind and mypy reports a misspelled
  kind or keyword. Every string option of the factories is a `Literal`
  type: `ElectrodeKind`, `SourceKind` and `KernelKind` (new, in
  `groundfield.api`), and `Backend`, `EarthInductiveModel`,
  `ConductorType`, `SoilCoupling`, `InductanceModel` where they are
  defined. mypy now finds all four typos of the review example (wrong
  kind, wrong keyword, `coupling_to_soil="galvanik"`, `kind="Current"`);
  it found none before. Test:
  `tests/test_api_factories.py::test_mypy_finds_the_four_typos`.
- **"Did you mean …?" errors (10.68):** an unknown electrode kind
  (`ValueError`), an unknown keyword of a kind (`TypeError`, naming the
  kinds that do have it), an unknown source kind (`ValueError`) and an
  unknown electrode name as a conductor end (`KeyError`) name the
  closest valid names. Tests:
  `tests/test_api_factories.py::test_unknown_electrode_kind_suggests_the_closest`,
  `::test_unknown_electrode_keyword_suggests_the_closest`,
  `::test_keyword_of_another_kind_is_explained`,
  `::test_unknown_source_kind_suggests_the_closest`,
  `::test_unknown_conductor_end_suggests_the_closest`.
- **Curated namespaces (10.72):** `gf.plot` (all plot helpers), `gf.safety`
  (touch and step voltages, EN 50522 limit) and `gf.fit` (vector fitting
  and the rho-f standard form) are new modules (`groundfield.plot`,
  `groundfield.safety`, `groundfield.fit`); `gf.io` groups the export.
  They hold the same objects as the flat names, which keep working;
  `dir(gf)` lists the namespaces. Test:
  `tests/test_package_namespace.py::test_namespace_is_an_importable_module`.
- **`create_engine(kernel=...)`:** chooses the field / reaction-matrix
  kernel (`"point"` or `"line"`, `KernelKind`). Test:
  `tests/test_api_factories.py::test_create_engine_accepts_the_point_kernel`.
- **`groundfield.utils`:** `suggest`, `did_you_mean`, `warn_deprecated`,
  `warn_inert`, `InertConfigurationWarning` (a `UserWarning`),
  `REMOVAL_VERSION` (`"2027.5"`). Deprecation and inert-configuration
  warnings point at the caller's line and are not emitted when
  groundfield's own code is the caller. Test:
  `tests/test_utils_helpers.py::test_helpers_stay_silent_for_groundfield_callers`.

### Changed

- **Faster `import groundfield` (10.72):** post-processing, IO,
  generators, the geo ingest, references, cross-engine validation and
  diagnostics are imported on first use (PEP 562); pandas, matplotlib,
  sympy and scipy are no longer loaded at startup. Median import time
  798 ms → 223 ms and 1169 → 346 loaded modules (15 fresh interpreters,
  shared 2-core sandbox). `__all__` is unchanged. Test:
  `tests/test_package_namespace.py::test_import_does_not_load_heavy_dependencies`.
- **Soil parameter names (10.69):** the model layer uses the literature
  symbols — `HomogeneousSoil.rho`, `SoilLayer.rho`, `SoilLayer.h`
  (`TwoLayerSoil.rho_1`, `rho_2`, `h_1` unchanged). The canonical world
  JSON writes `rho` / `h`; JSON written by 0.15.0 still loads. No
  numerical change. Golden: `gen_layout_builder` (`world/json` only).
  Test: `tests/test_soil_models.py::test_world_json_written_by_v0150_loads`.
- **`create_electrode` with an unknown keyword raises `TypeError`** (was a
  pydantic `ValidationError`), like the other factories, whose explicit
  signatures always raised `TypeError`.
- **`create_engine` forwards only the arguments you pass;** everything
  else takes the `Engine` default (single source of the defaults).
  `create_engine()` builds the same engine as `Engine()`. Test:
  `tests/test_api_factories.py::test_create_engine_forwards_only_explicit_arguments`.
- **Soil models validate assignment,** and `SoilModel` is a union
  discriminated by `kind` (inferred from the keys when a mapping omits
  it), so a soil is validated once and errors name the selected model.
- **`model_copy(update=...)` of a soil model rejects unknown keys** (a
  misspelled key was silently stored next to the fields and the copy
  kept the old value). Test:
  `tests/test_soil_models.py::test_model_copy_rejects_unknown_fields`.
- **`run_simulation(world, engine)`** is documented and tested as
  `world.solve(engine)`, with the same source-snapshot semantics. Test:
  `tests/test_api_factories.py::test_run_simulation_has_world_solve_semantics`.

### Fixed

- **Non-finite soil parameters (10.67):** `resistivity = inf`,
  `rho_1 = inf`, `rho_2 = inf`, `h_1 = inf`, an infinite layer thickness
  or `relative_permittivity` were accepted; an infinite resistivity then
  failed inside the solver with `LinAlgError: Singular matrix`. They are
  now rejected at construction and on assignment, naming the parameter.
  Tests: `tests/test_soil_models.py::test_soil_parameters_must_be_finite_and_positive`,
  `::test_assignment_is_validated`.
- **`MultiLayerSoil` structure errors (10.67):** an empty stack, a
  single layer with a finite thickness, a half-space above the bottom
  and a finite bottom layer each raise a message that names the layer
  and the fix; a one-layer stack with `h=None` stays a valid
  homogeneous half-space. Test:
  `tests/test_soil_models.py::test_multilayer_structure_errors_are_clear`.
- **Inert settings were silent (10.51 part, F57):** a non-default
  `BoundaryConditions` field set through the constructor,
  `create_world(boundary=...)` or a world JSON, and a non-default soil
  `relative_permittivity`, now warn with `InertConfigurationWarning`. No
  backend reads them: with every boundary field changed, all eight
  backends solve bit-identically, and ε_r = 1 and 81 give identical
  impedances (below 1 kHz the displacement current is at most
  ωε₀ε_rρ = 4.5·10⁻³ of the conduction current for ρ = 1000 Ω·m,
  ε_r = 81). `World.set_boundary_conditions` keeps its single warning.
  Tests: `tests/test_boundary_conditions.py::test_non_default_field_warns_on_construction`,
  `::test_boundary_conditions_are_inert_in_every_backend`,
  `tests/test_soil_models.py::test_non_default_relative_permittivity_warns`.
- **`annotations` leaked into the top-level namespace (10.72):**
  `gf.annotations` (the `__future__` feature object) is gone. Test:
  `tests/test_package_namespace.py::test_annotations_does_not_leak`.

### Deprecated

Old soil parameter names (10.69). They keep working as keyword, JSON key,
attribute and `model_copy(update=...)` key, warn with
`DeprecationWarning`, and will be removed in 2027.5 at the earliest.
Passing an old and its new name together raises. Type checkers already
report the old keywords (only the canonical names are part of the typed
signature); at run time they keep working.

| Deprecated | Replacement |
|---|---|
| `HomogeneousSoil(resistivity=...)`, `HomogeneousSoil.resistivity` | `rho` |
| `SoilLayer(resistivity=...)`, `SoilLayer.resistivity` | `rho` |
| `SoilLayer(thickness=...)`, `SoilLayer.thickness` | `h` |

Test: `tests/test_soil_models.py::test_old_keyword_warns_and_maps_to_canonical_field`.
