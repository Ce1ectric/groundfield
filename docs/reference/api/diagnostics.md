# Diagnostics

`groundfield.diagnostics` checks a world **before** a long solve: its
size, the discretisation quality, wires that are too close for the
thin-wire model, settings that have no effect, and — given a result —
truncated series. `diagnose(world, engine)` returns one aggregated
`DiagnosticsReport` with one line per cause (not per electrode); the
solve emits the same findings as warnings. The post-solve counterpart,
a comparison of several engines, is [Validation](validation.md).

| Function | Answers |
|---|---|
| `world_statistics(world)` | counts by kind, wire and conductor lengths, footprint |
| `expected_segments(world, engine)` | segments the discretiser will produce, per electrode and in total, the size of the nodal system and the engine's limit (`"limit"`) |
| `check_segment_resolution(world, engine)` | thin-wire ratio, electrodes smaller than a segment, size budget (strings) |
| `check_segment_budget(world, engine)` | raises `ResourceLimitError` above the engine's limit `Engine.max_segments` (default `HARD_LIMIT`; used by the solver) |
| `diagnose(world, engine, result=None)` | all of the above plus mixed local and projected coordinates, close wires, rod ends at a layer interface, inert settings, series truncation |

## Mathematical and physical context

The checks guard the assumptions of the thin-wire model (see
[Geometry](geometry.md)):

- **Coordinates.** A world whose points lie both within 10 km of the
  origin and beyond 1 000 km from it (horizontal distance) mixes local and
  projected coordinates: electrodes at UTM coordinates and a node given in
  a local system, almost always by mistake. It is solved as given, its two
  groups some 33 000 km apart (a return rod at local (80, 15) m for an
  outline and a rod at UTM coordinates: $Z_{in}$ +0.43 %; a PEN cable to a cabinet at local coordinates:
  a 33 118 km conductor; audit W2B-04). Category `coordinates`, a warning
  that names both groups and the conductors between them; the solve warns
  with `MixedCoordinatesWarning` too (user decision D5). A world that does
  not mix but whose horizontal extent exceeds 100 km is reported as
  information (two UTM zones, a zone prefix on some eastings only, a
  geometry in millimetres). See
  [Units and conventions](../units-and-conventions.md#coordinates).

- **Thin-wire ratio.** The self term of a segment of length $L$ and
  radius $a$ behaves like $2\ln(L/a)/L$ and loses dominance as $L$
  approaches $a$; the check asks for $L/a \ge 5$ (`MIN_THINWIRE_RATIO`).
- **Resolution.** An electrode feature shorter than one segment is
  represented only by the discretiser's floor (one segment per wire,
  eight per ring).
- **Proximity.** Two wires of different owners closer than five times
  the sum of their radii violate the point-source assumption that a
  source sees another wire only through its midpoint; the exact
  line-source kernel stays physical there.
- **Interface contact.** The thin-wire model has no end face. A rod
  whose head or tip lies within one wire radius of a layer interface is
  either entirely on one side of it (end at the interface) or touches the
  other layer through a part shorter than its radius; the solid rod is
  continuous in between. Against an axisymmetric model of the solid rod
  (4.5 m, $a$ = 8 mm, 1000/30 Ω·m) the thin wire reads +21 % with its tip
  at the interface and −4.6 % 1 mm past it. A horizontal wire within a few
  wire radii of a contrasting interface carries its current on the axis
  while the solid wire's current crowds to the near side: +3.6 % at two
  radii over a layer 33 times more conductive (audits W4A-07, W4B-02; the
  reach per contrast is tabulated under `InterfaceContactWarning` in the
  [warnings reference](../warnings.md)). Category `interface_contact`, also
  `InterfaceContactWarning` at the solve.
- **Size.** The dense reaction matrix of $N$ segments has $N^2$
  entries ($8N^2$ bytes, 3.0 GiB at $N$ = 20 000); the work per
  excitation grows like $N^2$ and the direct solve like $N^3$. The
  dense nodal system of $n$ unknowns (current owners, nodes and
  branches, every conductor sub-piece included) takes $16n^2$ bytes.
  At or above the engine's notice size the report informs that a
  solve takes minutes (an *info* finding; every `Engine.solve`, with any
  backend, emits it as a `UserWarning` — `solve_mutual_matrix` and
  `solve_mutual_field` do not); above the engine's limit the solver
  refuses to start with `ResourceLimitError`, before anything of that
  size is allocated (every segment backend and `solve_mutual_*`).
  Both are set per engine and count both sizes: the limit with
  `Engine(max_segments=...)`, the notice with
  `Engine(warn_segments=...)` (positive integers; user decisions Q17 and
  Q31 of 2026.11). An engine that leaves them at `None` uses the module
  constants `HARD_LIMIT` (20 000) and `SOFT_LIMIT` (5 000), read at
  solve time. The two are not checked against each other: a notice
  above the hard limit never appears, because such a world is refused
  first. `expected_segments(...)["limit"]`,
  `report.context["max_segments"]` and `FieldResult.metadata
  ["max_segments"]` show the limit that applies,
  `report.context["warn_segments"]` the notice size; the provenance
  block of a result keeps both requests.

The segment counts follow the discretisation of the image and MoM
families; for `fem` they do not predict the cost. `fem` does not use those
segments: its problem is the mesh of each driven cluster (2 400 nodes at
the default resolution in homogeneous soil, growing with the depth of the
layer stack to about 4 100 at 30 m and 6 200 at 1 km, independent of
`segment_length`), so its
`Engine.solve` still emits the notice from the segment count but does not
refuse a world above `Engine.max_segments`.

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
gf.create_electrode(world, "rod", name="a", position=(0.0, 0.0, 0.0), length=3.0, wire_radius=0.008)
gf.create_electrode(world, "rod", name="b", position=(0.018, 0.0, 0.0), length=3.0, wire_radius=0.008)
gf.create_source(world, attached_to="a", magnitude=1.0)
engine = gf.create_engine(segment_length=0.5)

budget = gf.expected_segments(world, engine)
report = gf.diagnose(world, engine)
print(report)    # proximity: 'a'–'b' close, axis distance 18 mm (clearance 2 mm) ...

assert budget["total"] == 12                      # two rods of six 0.5 m segments
assert "proximity" in report.categories()
```

The limit is a property of the engine: a smaller one refuses this world
before the solve allocates anything, a larger one admits worlds beyond
`HARD_LIMIT`.

```python
from groundfield.diagnostics import HARD_LIMIT, ResourceLimitError, check_segment_budget

try:
    check_segment_budget(world, gf.create_engine(segment_length=0.5, max_segments=10))
except ResourceLimitError as error:
    print(error)    # ... discretises into 12 segments ..., more than Engine.max_segments = 10 ...
print(budget["limit"], gf.expected_segments(world, gf.create_engine(max_segments=50_000))["limit"])
```

<!-- doctest-check
assert budget["limit"] == HARD_LIMIT == 20_000
assert gf.expected_segments(world, gf.create_engine(max_segments=50_000))["limit"] == 50_000
-->

The notice "solve time may run from seconds to minutes" works the same way:
`SOFT_LIMIT` is the default, `Engine(warn_segments=...)` moves it for one
engine, for instance down to a size that is already slow on a small
machine.

```python
from groundfield.diagnostics import SOFT_LIMIT

early = gf.Engine(segment_length=0.5, warn_segments=10)
notice = gf.diagnose(world, early).by_category("budget")
print(notice[0].severity, notice[0].message)    # info predicted total segment count 12 is at or above Engine.warn_segments = 10 — ...
print(gf.diagnose(world, engine).context["warn_segments"])    # 5000: SOFT_LIMIT
```

<!-- doctest-check
assert [d.severity for d in notice] == ["info"] and notice[0].data["limit"] == 10
assert gf.diagnose(world, engine).context["warn_segments"] == SOFT_LIMIT == 5_000
assert not gf.diagnose(world, engine).by_category("budget")
-->

## API reference

::: groundfield.diagnostics
