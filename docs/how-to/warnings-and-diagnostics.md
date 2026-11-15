# Read and act on warnings

groundfield warns when a result is computed but may not mean what you
think: a discretisation that is too coarse somewhere, a model applied
outside its validity, a setting that has no effect, a deprecated name.
Each warning names its cause and a remedy. This guide shows how to check
a world *before* a long run with `gf.diagnose`, lists the warnings you
will meet most often, and shows how to handle warnings in scripts and
tests. Every warning and error class is described in the
[warnings catalogue](../reference/warnings.md).

## Check a world before you solve

`gf.diagnose(world, engine)` collects the pre-solve checks in one report:
resolution (features shorter than the segment length), segment budget,
thin wires, wires of different electrodes that are too close or
interpenetrate, inductive conductors drawn on top of each other, and
settings that have no effect. Pass `result=` as well
to add post-solve checks such as a truncated image series.

```python
import warnings

import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="rod_a", position=(0.0, 0.0, 0.0), length=3.0, wire_radius=0.008)
gf.create_electrode(world, "rod", name="rod_b", position=(0.018, 0.0, 0.0), length=3.0, wire_radius=0.008)
gf.create_conductor(world, name="bond", start="rod_a", end="rod_b")  # two rods 2 mm apart
gf.create_electrode(world, "rod", name="short_rod", position=(10.0, 0.0, 0.5), length=0.4)
gf.create_conductor(world, name="link", start="rod_a", end="short_rod")
gf.create_source(world, attached_to="rod_a", magnitude=1.0)

report = gf.diagnose(world, gf.create_engine(segment_length=1.0, kernel="point"))
print(report)
for finding in report:
    print(finding.category, finding.severity, finding.subjects)
```

<!-- doctest-check
assert {f.category for f in report} >= {"resolution", "proximity"}
assert any(f.severity == "warning" for f in report if f.category == "proximity")
-->

The report prints one line per cause. Acting on it here means: refine the
segment length (or accept the coarse short rod), and either merge the two
rods into one electrode or use the default line-source kernel, for which
close wires are only information:

```python
report_line = gf.diagnose(world, gf.create_engine(segment_length=0.2))
print(report_line or "no findings")
```

<!-- doctest-check
assert all(f.severity != "warning" for f in report_line)
-->

`report.warn()` emits the findings as one `DiagnosticsWarning`, for
pipelines that collect warnings.

Insulated cables with an inductance model are checked as well: two
conductors drawn beside each other closer than touching over a stretch
(category `overlap`, a warning) are evaluated by the inductance matrix as
touching wires, so their coupling follows their radii instead of the drawn
spacing. A typical cause is a second cable or a service drop drawn on the
axis of another:

```python
cables = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(cables, "rod", name="a", position=(0.0, 0.0, 0.5), length=1.5)
gf.create_electrode(cables, "rod", name="b", position=(50.0, 0.0, 0.5), length=1.5)
for name, radius in (("cable_1", 0.006), ("cable_2", 0.004)):
    gf.create_conductor(
        cables, name=name, start="a", end="b", wire_radius=radius,
        coupling_to_soil="isolated", inductance_model="neumann",
    )
overlap = [f for f in gf.diagnose(cables) if f.category == "overlap"]
print(overlap[0].message)   # 'cable_1'–'cable_2' beside each other over 50 m, ...
```

<!-- doctest-check
assert len(overlap) == 1 and overlap[0].severity == "warning"
assert abs(overlap[0].data["overlap_m"] - 50.0) < 1e-6
-->

Draw the cables at their real spacing (a few centimetres in one trench),
or model a concentric cable with `conductor_type="cable_shield"` around
its core, which is not an overlap. The stretch must exceed ten touching
distances and 5 % of the shorter conductor, so a branch leaving its main
at more than about half a degree, a crossing and a T-contact are not
reported, and neither are cables drawn exactly at the touching distance
(within 1e-9 of it: round-off, audit W3B-03). The generated networks
report none: the service drops that leave one tap in one direction, the
drops of houses behind each other in the radial networks and in star
networks such as the 10-house star (five drops from the substation), lie
side by side at the touching distance since decision D9, so that only
neighbours touch. Up to then they lay on one axis and every pair was
reported (radial networks of 5 / 10 / 80 / 200 houses: 10 / 20 / 31 /
247 pairs, the 10-house star 10, the 80-house star with commercial sites
2; audit W3B-05).

## The warnings you will meet most often

| Warning | Meaning | What to do |
|---|---|---|
| `UserWarning` "resolution: … shorter than segment_length" | an electrode feature is shorter than the segment length; the discretiser falls back to its minimum | refine `segment_length`, or accept it for small parts that do not matter |
| `UserWarning` "… above the layer interface … less than 2×segment_length" | a wire lies close to the soil-layer interface | refine `segment_length` near the interface |
| `ShallowSegmentWarning` | segments longer than four times their burial depth, with the point kernel or the `mom` family | use the image family's default line kernel, or shorter segments |
| `SeriesTruncationWarning` | the two-layer image series stopped at `image_max_terms` before reaching its tolerance (strong contrast, $\lvert K\rvert \to 1$) | raise `Engine(image_max_terms=...)` |
| `UniformCurrentFieldWarning` | touch or step voltages from the image backend next to grids, meshes, stars, strips or bonded clusters | use `backend="mom"` for field quantities (tutorial [T08](../tutorials/T08_touch_and_step_assessment.ipynb)) |
| `FemReductionWarning`, `ExperimentalBackendWarning` | the experimental `fem` backend reduces the world to an equivalent hemisphere that does not represent it | use `mom` or `mom_sommerfeld` |
| `BackendCapabilityWarning` | the backend solves the world with a simplification (for example lumped conductors) | choose a backend that supports the feature |
| `InertConfigurationWarning` | a setting that no backend reads (boundary conditions, relative permittivity) was changed | remove it; it has no effect below 1 kHz |
| `RhoFStandardQualityWarning`, `VectorFitQualityWarning` | a rho-f fit misses its samples by more than the tolerance | narrow the range, add poles, or fit per resistivity (tutorial [T04](../tutorials/T04_frequency_response_and_rho_f.ipynb)) |
| `NonPassiveFitWarning` | an exported formula has a negative resistance somewhere in the band | refit; export with `require_passive=True` to refuse such formulas |
| `CrossSectionUnitWarning` | a cross section above 0.01 m² — probably mm² passed as m² | pass m² (`50e-6` for 50 mm²) |
| `DeprecationWarning` | a name or parameter is deprecated | switch to the replacement it names; deprecated names are removed in 2027.5 at the earliest |

Errors stop the solve before anything expensive happens:
`BackendCapabilityError` (the backend cannot represent the world; the
message names backends that can), `ResourceLimitError` (more segments
or nodal unknowns than the engine's limit, `Engine(max_segments=...)`,
by default `groundfield.diagnostics.HARD_LIMIT`; the notice that a solve
may take minutes moves with `Engine(warn_segments=...)`), and `ValueError` /
`KeyError` for invalid input with the closest valid names.

## Handle warnings in scripts and tests

Warnings point at the line of *your* code that caused them. Do not
silence them globally; catch the ones you have understood, where they
occur:

```python
engine = gf.create_engine(segment_length=1.0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = engine.solve(world)
for w in caught:
    print(f"{w.category.__name__}: {str(w.message)[:90]} ...")
```

<!-- doctest-check
assert any("resolution" in str(w.message) for w in caught)
-->

<!-- doctest-allow-warnings: DeprecationWarning -->

```python
with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)  # make deprecations fail here
    try:
        gf.HomogeneousSoil(resistivity=100.0)  # the pre-2026.11 name
    except DeprecationWarning as exc:
        print("DeprecationWarning:", exc)
```

In a test suite, turn groundfield's deprecations into failures so that an
upgrade cannot hide them, e.g. with pytest in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
filterwarnings = ["error::DeprecationWarning"]
```
