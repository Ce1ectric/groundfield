# Diagnostics

The :mod:`groundfield.diagnostics` module provides **pre-solve**
structural diagnostics for a :class:`World`. It is the AP1
counterpart to :mod:`groundfield.validation` (which is the
**post-solve** cross-engine consistency check).

## Why pre-solve diagnostics?

For an AP1-grade TN-Ortsnetz with several hundred electrodes a
mistake in the geometry — an electrode placed at the wrong
coordinate, a missing conductor, or a segmentation budget that
silently triples the wall-clock time — should be caught **before**
kicking off a multi-minute solver run. The three helpers below
each address one common failure mode:

| Helper                          | What it answers                                                         |
|---------------------------------|-------------------------------------------------------------------------|
| `world_statistics`              | "How big is this thing? Counts per kind, total wire length, footprint." |
| `expected_segments`             | "How many point-source segments will the discretiser produce?"          |
| `check_segment_resolution`      | "Will the resolution be good enough? Will memory / time blow up?"       |

## Example — full pre-flight check

```python
import groundfield as gf
from groundfield.generators import TnNetworkGenerator, TnNetworkConfig, PenConfig

cfg = TnNetworkConfig(
    soil=gf.HomogeneousSoilSpec(resistivity=100.0),
    building_counts={"residential": 30},
    pen=PenConfig(segment_length_m=None),
)
world = TnNetworkGenerator(cfg, seed=42).build()
engine = gf.create_engine(backend="image", segment_length=0.5)

# 1. Structural snapshot (machine-readable; pretty-print as needed).
stats = gf.world_statistics(world)
print(f"electrodes:  {stats['n_electrodes']}  "
      f"by kind: {stats['n_electrodes_by_kind']}")
print(f"footprint:   {stats['footprint_area_m2']:.0f} m^2")
print(f"wire length: {stats['total_electrode_wire_length_m']:.1f} m  "
      f"(electrodes) + "
      f"{stats['total_conductor_length_m']:.1f} m (conductors)")

# 2. Segmentation budget.
budget = gf.expected_segments(world, engine)
print(f"predicted total segments: {budget['total']}  "
      f"(electrodes {budget['electrode_total']}, "
      f"conductors {budget['conductor_total']})")

# 3. Quality-of-discretisation warnings (empty list = all good).
for msg in gf.check_segment_resolution(world, engine):
    print(f"  WARN: {msg}")
```

## `expected_segments` — exactness

The prediction is **bit-exact** for the image-family backends —
``image``, ``image_2layer``, ``image_nlayer``, ``mom``,
``mom_sommerfeld``, ``cim`` and ``bem``. It mirrors
:mod:`groundfield.solver.image`'s discretiser conventions:

- rod: $n = \max(1, \lceil L / \Delta s \rceil)$
- ring: $n = \max(8, \lceil 2 \pi r / \Delta s \rceil)$
- strip: $n = \max(1, \lceil L / \Delta s \rceil)$
- mesh / grid_mesh: per-wire
  $n = \max(1, \lceil d_\text{axis} / \Delta s \rceil)$,
  summed over both wire directions
- distributed conductor: $n = \lceil L / \Delta s_c \rceil$
  with $\Delta s_c = $ ``conductor.discretize_segment_length``

The FEM backend (:mod:`groundfield.solver.fem`) uses an
axisymmetric volume mesh that is not parameterised by
``segment_length``; the prediction is **not informative** for
FEM.

## `check_segment_resolution` — heuristics

Three categories of warning are surfaced:

- **Thin-wire ratio** $\Delta s / r_\text{wire} \ge 5$ on every
  electrode and on every distributed conductor. Below this, the
  thin-wire average-potential self-action becomes biased.
- **Electrode smaller than one segment** — the smallest geometric
  dimension of an electrode (rod length, ring perimeter, strip
  length, mesh wire length) must be at least one
  ``segment_length``; otherwise the discretiser falls back to
  its lower floor.
- **Total segment-count budget** at a soft threshold (5 000
  segments — solve time runs from seconds to minutes) and a
  hard threshold (20 000 segments — the dense-system $O(N^2)$
  memory and $O(N^3)$ solve scaling becomes painful).

The function never raises — use the returned list to inform the
user. An empty list means *no concerns detected*.

## API reference

::: groundfield.diagnostics
