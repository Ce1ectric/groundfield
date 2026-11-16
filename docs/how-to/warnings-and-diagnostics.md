# Read and act on warnings

This guide checks a world for problems before a long run with
`gf.diagnose`, explains the warnings you will meet most often and what to
do about each, and shows how to handle warnings in scripts and tests. Read
it when a solve warns, or before you start a large study.

groundfield warns when a result is computed but may not mean what you
think: a discretisation that is too coarse somewhere, a model applied
outside its validity, a setting that has no effect, a deprecated name.
Each warning names its cause and a remedy. Every warning and error class,
with its import path, is described in the
[warnings catalogue](../reference/warnings.md).

## 1. Check a world before you solve

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

**What you get:** with the point kernel a `resolution` finding for the
short rod and a `proximity` warning for the two rods 2 mm apart; with the
default line kernel and 0.2 m segments no warning is left (the hidden
checks assert both). `report.warn()` emits the findings as one
`DiagnosticsWarning`, for pipelines that collect warnings.

## 2. Check insulated cables for overlaps

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
its core, which is not an overlap.

What is reported, and what is not:

- The stretch must exceed ten touching distances and 5 % of the shorter
  conductor, so a branch leaving its main at more than about half a
  degree, a crossing and a T-contact are not reported.
- Cables drawn exactly at the touching distance are not reported either:
  within 1e-9 of it, or within 16 units in the last place of the
  coordinates where that is more (round-off, also at projected
  coordinates).
- The generated networks report none, also with placement jitter (star
  and radial networks of 5 … 200 houses, `jitter_m` 0 … 0.1 m). The
  service drops that leave one tap in one direction — the drops of houses
  behind each other in the radial networks and in star networks such as
  the 10-house star (five drops from the substation) — lie side by side at
  the touching distance, so that only neighbours touch. Drops that are not
  exactly parallel (jitter) are laid along one common normal, and a drop
  that would lie closer than touching to a drop of a row joins the row.

With a bare PEN (`coupling_to_soil="galvanic"`) the drops of a row are
close wires (`proximity` "close", 1 … 4 cm apart), and a drop of a row
starts up to (n − 1)/2 touching distances off its tap (2 cm in a row of
five), next to the wire it leaves there. Without an engine `diagnose()`
warns about those starts as "interpenetrating" against the substation ring
(the 10-house radial network: 9 pairs at axis distances of 0.5 … 5.0 mm)
and about the neighbours of a row as "close"; with the default engine (the
image family's line kernel, which integrates near pairs) all of them are
information.

??? note "Background and audit trail"

    - The tolerance at the touching distance (1e-9, or 16 units in the last
      place) dates from audits W3B-03 and W3R2B-02.
    - The drops of a row lie side by side at the touching distance since
      decision D9. Up to D9 they lay on one axis and every pair was
      reported (radial networks of 5 / 10 / 80 / 200 houses: 10 / 20 / 31 /
      247 pairs, the 10-house star 10, the 80-house star with commercial
      sites 2; audit W3B-05). Up to audit W3R2B-01 a jitter of 1 nm put
      them back on one axis (the 10-house radial network: 19 pairs); since
      then jittered drops are laid along one common normal and a drop too
      close to a row joins the row.
    - With the drops on one axis (before D9) the bare-PEN case gave 5
      "interpenetrating" pairs at 5.0 mm in the 10-house radial network.
      That the default engine reports all of these pairs as information is
      the finding of audit W3R2B-07.

## 3. Know the warnings you will meet most often

| Warning | Meaning | What to do |
|---|---|---|
| `UserWarning` "resolution: … shorter than segment_length" | an electrode feature is shorter than the segment length | refine `segment_length`, or accept it for small parts |
| `UserWarning` "image_2layer: deepest segment ends … above the layer interface" | point kernel: the deepest segment ends less than two segment lengths above the interface | refine near the interface, or use the line kernel |
| `ShallowSegmentWarning` | segments longer than four times their depth (point kernel, `mom` family) | the default line kernel, or shorter segments |
| `SeriesTruncationWarning` | the two-layer image series stopped at `image_max_terms` | raise `Engine(image_max_terms=...)` |
| `UniformCurrentFieldWarning` | touch or step voltages of the image family near grids, stars, strips, bonded clusters | `backend="mom"` for field quantities |
| `ExperimentalBackendWarning` | the first solve with the experimental `fem` backend | use `fem` as a reference next to a production engine |
| `InterfaceContactWarning` | a rod end or a horizontal wire very close to a layer interface | keep wires a few radii from the interface |
| `BackendCapabilityWarning` | the backend simplifies the world (for example lumped conductors) | choose a backend that supports the feature |
| `InertConfigurationWarning` | a setting that no backend reads was changed | remove it |
| `RhoFStandardQualityWarning`, `VectorFitQualityWarning` | a rho-f fit misses its samples by more than the tolerance | narrow the range, add poles, or fit per resistivity |
| `NonPassiveFitWarning` | an exported formula has a negative resistance in the band | refit; `require_passive=True` refuses such formulas |
| `CrossSectionUnitWarning` | a cross section above 0.01 m², probably mm² passed as m² | pass m² (`50e-6` for 50 mm²) |
| `DeprecationWarning` | a name or parameter is deprecated | switch to the replacement it names |

Details of some rows:

- **The image-series warnings.** `SeriesTruncationWarning` appears for
  strong contrasts, $\lvert K\rvert \to 1$ with the reflection coefficient
  $K = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)$, where the series needs more
  terms than `image_max_terms` to reach its tolerance. The
  "deepest segment ends" warning comes from `image_2layer` with
  `kernel="point"`.
- **`UniformCurrentFieldWarning`**: the image family carries one uniform
  current per straight wire, which is right for impedances but not for
  the local field next to grids, meshes, stars, strips or bonded clusters.
  Tutorial [T08](../tutorials/T08_touch_and_step_assessment.ipynb) shows
  the assessment with `mom`.
- **`ExperimentalBackendWarning`** is emitted once per process; filter the
  category when you use `fem` on purpose. `FemReductionWarning`, the
  warning of the former equivalent-hemisphere reduction of `fem`, is no
  longer emitted since 2026.11.
- **`InterfaceContactWarning`**: a rod end within one wire radius of a
  layer interface, or a horizontal wire within a few wire radii of a
  contrasting one — 3.9 radii over a layer 33 times more conductive,
  2.2 radii under one 33 times more resistive, one radius from a contrast
  of 1.3 for a wire the interface cuts. The thin-wire model deviates from
  the solid wire by 1 % or more there. The image family, `mom`,
  `mom_sommerfeld` and the 3-D `fem` warn, the axisymmetric `fem` does
  not: check a ring with `fem`'s axisymmetric model (the reach of the
  interface is tabulated in the [warnings reference](../reference/warnings.md)).
- **`InertConfigurationWarning`** covers boundary conditions and the
  relative permittivity; neither has an effect below 1 kHz.
- **The rho-f warnings**: tutorial
  [T04](../tutorials/T04_frequency_response_and_rho_f.ipynb) fits and
  checks rho-f models.
- **`DeprecationWarning`**: deprecated names are removed in 2027.5 at the
  earliest.

Errors stop the solve before anything expensive happens:
`BackendCapabilityError` (the backend cannot represent the world; the
message names backends that can), `ResourceLimitError` (more segments
or nodal unknowns than the engine's limit, `Engine(max_segments=...)`,
by default `groundfield.diagnostics.HARD_LIMIT`; the notice that a solve
may take minutes moves with `Engine(warn_segments=...)`), and `ValueError` /
`KeyError` for invalid input with the closest valid names.

## 4. Handle warnings in scripts and tests

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

## See also

- [Warnings and errors](../reference/warnings.md): every class, when it is
  emitted and what to do.
- [Known limitations](../reference/known-limitations.md): the limits the
  warnings guard.
- [Diagnostics API](../reference/api/diagnostics.md): `diagnose`, the
  report and its categories.
