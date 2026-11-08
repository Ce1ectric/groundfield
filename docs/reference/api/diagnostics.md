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
| `expected_segments(world, engine)` | segments the discretiser will produce, per electrode and in total |
| `check_segment_resolution(world, engine)` | thin-wire ratio, electrodes smaller than a segment, size budget (strings) |
| `check_segment_budget(world, engine)` | raises `ResourceLimitError` above `HARD_LIMIT` (used by the solver) |
| `diagnose(world, engine, result=None)` | all of the above plus close wires, inert settings, series truncation |

## Mathematical and physical context

The checks guard the assumptions of the thin-wire model (see
[Geometry](geometry.md)):

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
- **Size.** The dense reaction matrix of $N$ segments has $N^2$
  entries; the work per excitation grows like $N^2$ and the direct
  solve like $N^3$. Above `SOFT_LIMIT` (5 000 segments) the report
  warns, above `HARD_LIMIT` (20 000) the solver refuses to start
  (`groundfield.diagnostics.HARD_LIMIT` can be raised on purpose).

The segment counts follow the discretisation of the image and MoM
families; for `fem` they do not predict the cost.

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

## API reference

::: groundfield.diagnostics
