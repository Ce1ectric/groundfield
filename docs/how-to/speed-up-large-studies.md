# Speed up large studies

This guide makes large solves — networks of many houses, fine
discretisations, inductive conductors at many frequencies — fast enough,
with the levers in the order in which they usually matter, and shows how to
measure each one on your own machine. Read it before a study whose single
solves take more than a few seconds.

The cost of a solve grows with the number of wire segments $N$ (the
reaction matrix has $N^2$ entries and the solve is a dense $N \times N$
system), with the number of inductive conductor segments, and with the
earth-return model. Measured run times of this release are on the
[performance page](../explanation/performance.md).

**Prerequisites:** the basic workflow of tutorial
[T01](../tutorials/T01_first_grounding_impedance.ipynb). All blocks of this
page build on the test network of step 1 (a ring station and six house
grids joined by a PEN cable).

## 1. Count the segments before you solve

`gf.expected_segments(world, engine)` predicts the discretisation exactly
as the solver will do it, per electrode and in total, without solving.
`gf.world_statistics(world)` summarises the geometry.

```python
import time

import numpy as np

import groundfield as gf
from groundfield.diagnostics import HARD_LIMIT


def network(n_houses=6, inductive=False, rho=100.0):
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=rho))
    gf.create_electrode(world, "ring", name="station", center=(0.0, 0.0, 0.6), radius=2.0)
    previous = "station"
    for k in range(n_houses):
        gf.create_electrode(
            world, "grid_mesh", name=f"house_{k}", corner=(30.0 * (k + 1) - 5.0, 8.0, 0.8), size=(10.0, 10.0), n_x=1, n_y=1
        )
        gf.create_conductor(
            world,
            name=f"pen_{k}",
            start=previous,
            end=f"house_{k}",
            conductor_type="pen",
            resistivity=2.82e-8,
            cross_section=50e-6,
            coupling_to_soil="isolated",
            inductance_model="neumann" if inductive else None,
            discretize_segment_length=5.0 if inductive else None,
        )
        previous = f"house_{k}"
    gf.create_source(world, attached_to="station", magnitude=1.0)
    return world


world = network()
for ds in (2.0, 1.0, 0.5):
    count = gf.expected_segments(world, gf.create_engine(segment_length=ds))
    print(f"segment_length {ds:3.1f} m: {count['total']:5d} segments")
print("hard limit:", HARD_LIMIT)
```

<!-- doctest-check
counts = [gf.expected_segments(world, gf.create_engine(segment_length=ds))["total"] for ds in (2.0, 1.0, 0.5)]
assert counts[0] < counts[1] < counts[2]
-->

**What you get:** the segment count per segment length and the limit it
is checked against. Halving the segment length doubles $N$ and roughly
quadruples the matrix work. A world above the engine's limit is rejected with
`ResourceLimitError` before anything large is allocated.

### Raise or lower the resource limit

The limit bounds the number of leakage segments and the size of the dense
nodal system (current owners, nodes and branches, every conductor
sub-piece included). It belongs to the engine: `max_segments=None` (the
default) uses the module constant `HARD_LIMIT` (20 000), a positive
integer sets the limit for this engine only. The dense reaction matrix
takes $8N^2$ bytes, so check the memory of the machine before you raise
the limit — 40 000 segments need 12 GiB for the matrix alone. A lower
limit is a cheap guard for batch runs: a world that a unit error has
blown up fails at once instead of filling the memory. Every segment
backend, `solve_mutual_matrix` and `solve_mutual_field` honour the
limit, `gf.diagnose` reports against it, and the result records it:

```python
from groundfield.diagnostics import ResourceLimitError, check_segment_budget

guarded = gf.create_engine(segment_length=0.5, max_segments=200)
try:
    check_segment_budget(world, guarded)  # what guarded.solve(world) checks first
except ResourceLimitError as error:
    print("refused:", str(error).split(" (")[0])
large = gf.create_engine(segment_length=0.5, max_segments=50_000)
print("limit:", gf.expected_segments(world, large)["limit"])
result = large.solve(world)
print("checked against:", result.metadata["max_segments"])
```

<!-- doctest-check
assert gf.expected_segments(world, guarded)["total"] > 200
assert result.metadata["max_segments"] == 50_000
-->

Changing `groundfield.diagnostics.HARD_LIMIT` changes the default of
every engine of the process; `max_segments` is the per-engine setting.

### Move the size notice

Below the limit, every `Engine.solve` at or above a second size emits a
`UserWarning` that the solve time may run from seconds to minutes
(`"budget: ..."`, an *info* finding of `gf.diagnose`); `solve_mutual_matrix`
and `solve_mutual_field` do not emit it, and for the experimental `fem` the
segment count is not its problem size (the mesh of each cluster is). That
size belongs
to the engine as well: `warn_segments=None` (the default) uses the module
constant `SOFT_LIMIT` (5 000), a positive integer sets it for this engine
only and is counted like the limit (segments and nodal unknowns). On a
workstation that solves 10 000 segments quickly, raise it together with
`max_segments`; on a small machine, lower it. The two are not checked
against each other: a notice above the hard limit never appears, because
the world is refused first.

```python
quiet = gf.Engine(segment_length=0.5, max_segments=50_000, warn_segments=12_000)
early = gf.Engine(segment_length=0.5, warn_segments=400)
for label, engine in (("quiet", quiet), ("early", early)):
    notices = [d for d in gf.diagnose(world, engine, proximity=False).by_category("budget")]
    print(label, engine.warn_segments, [d.severity for d in notices])
```

<!-- doctest-check
assert not gf.diagnose(world, quiet, proximity=False).by_category("budget")
assert [d.severity for d in gf.diagnose(world, early, proximity=False).by_category("budget")] == ["info"]
-->

## 2. Choose the segment length from a convergence study

Use the coarsest segment length that is converged for the quantity you
need — tutorial [T09](../tutorials/T09_choosing_an_engine_and_convergence.ipynb)
shows how. With the default line-source kernel, network impedances are
usually converged at 0.5–1 m; compare two values on a representative
world before a large run:

```python
z = {ds: gf.create_engine(segment_length=ds).solve(world).input_impedance()[0].real for ds in (1.0, 0.5)}
print(f"1.0 m: {z[1.0]:.5f} Ohm, 0.5 m: {z[0.5]:.5f} Ohm, change {z[0.5] / z[1.0] - 1:+.2e}")
```

<!-- doctest-check
assert abs(z[0.5] / z[1.0] - 1.0) < 1e-3
-->

**How to check it:** the change printed above is the convergence of the
input impedance; the hidden check of this page asserts that it is below
1e-3 for this network.

## 3. Kernel and compiled extension

The exact line-source kernel (default of the image family) costs a few
times the point kernel per solve, but it is converged at much coarser
segments, so it is usually the faster way to a given accuracy. Install the
`fast` extra (`numba`) for compiled kernels with identical results.
Measure both kernels on your machine:

```python
for kernel in (None, "point"):
    engine = gf.create_engine(segment_length=0.5, kernel=kernel)
    start = time.perf_counter()
    result = engine.solve(world)
    elapsed = time.perf_counter() - start
    print(f"kernel {result.metadata['kernel']:5s}: {elapsed:.3f} s, Z_in = {result.input_impedance()[0].real:.5f} Ohm")
```

**What you get:** the time and the input impedance per kernel;
`result.metadata["kernel"]` records the kernel a result was solved with.
On this network the point kernel reads 0.6 % below the line kernel (on
single rods, rings and strips 1–2 %, at every segment length; see
[Kernels](../explanation/kernels.md)). In a fresh process the first
line-kernel solve also compiles the numba kernels, about a second once;
the earlier blocks of this page have done that already, so time a second
solve when you run the block on its own.

## 4. Many excitations: one matrix instead of many solves

When you need the response to an injection at *each* of many earth
electrodes — the mutual grounding-impedance matrix $Z_G$ of a site —
`gf.solve_mutual_matrix` assembles the reaction matrix once and solves
all excitations with it, instead of one full solve per electrode. Its
off-diagonal is the Galerkin reaction of the leakage distributions,
which is reciprocal and reproduces a solve of the clusters joined by
branches; probe points are not needed (`offdiag="probe"` samples the
potential at points instead). It expects a world with the earth electrodes
only (no cables between them):

```python
electrodes_only = network()
electrodes_only.conductors.clear()
electrodes_only.sources.clear()
anchors = [e.name for e in electrodes_only.electrodes]
engine = gf.create_engine(segment_length=1.0)

start = time.perf_counter()
z_g = gf.solve_mutual_matrix(electrodes_only, engine, anchors)
t_matrix = time.perf_counter() - start

start = time.perf_counter()
diagonal = []
for name in anchors:
    single = electrodes_only.model_copy(deep=True)
    gf.create_source(single, attached_to=name, magnitude=1.0)
    diagonal.append(engine.solve(single).cluster_impedance(name)[0])
t_solves = time.perf_counter() - start
print(f"{len(anchors)} excitations: one matrix {t_matrix:.2f} s, {len(anchors)} solves {t_solves:.2f} s")
print("largest difference on the diagonal:", f"{np.max(np.abs(np.diag(z_g) - diagonal)):.1e} Ohm")
```

<!-- doctest-check
assert np.max(np.abs(np.diag(z_g) - np.array(diagonal))) < 1e-9
assert np.allclose(z_g, z_g.T)
-->

**How to check it:** the diagonal of $Z_G$ equals the cluster impedances
of the single solves (the printed difference; the hidden check asserts
below 1e-9 Ω), and $Z_G$ is symmetric.

## 5. Frequencies and the earth-return model

- Put all frequencies into **one** engine: the soil problem is assembled
  once, and the earth-return correction of inductive conductors is built
  for the whole frequency set at once.
- The Sommerfeld correction of inductive conductors is interpolated
  from tables per pair of depth groups by default
  (`sommerfeld_assembly="tabulated"`) instead of integrated per segment
  pair (`"exact"`). The two agree to far below the accuracy of the model,
  and the tables are the closer of the two to a converged reference;
  compare them on your network:

```python
inductive = network(inductive=True)
frequencies = [50.0, 250.0, 550.0, 1000.0]
z_assembly = {}
for assembly in ("exact", "tabulated"):
    engine = gf.create_engine(segment_length=1.0, frequencies=frequencies, sommerfeld_assembly=assembly)
    start = time.perf_counter()
    z_assembly[assembly] = np.array(engine.solve(inductive).input_impedance())
    print(f"{assembly:9s}: {time.perf_counter() - start:.2f} s, Z_in(1 kHz) = {z_assembly[assembly][-1]:.6f} Ohm")
print(f"largest relative difference: {np.max(np.abs(z_assembly['tabulated'] / z_assembly['exact'] - 1)):.1e}")
```

<!-- doctest-check
assert np.max(np.abs(z_assembly["tabulated"] / z_assembly["exact"] - 1.0)) < 1e-6
-->

The difference in run time grows with the number of inductive segments
(on an 8-house TN network with an inductive PEN, 78 inductive branches
and four frequencies: 16.6 s CPU per-pair, 1.0 s tabulated).
`carson_series` is cheaper still, but it is a long-parallel-wire
approximation: for cables that run at an angle to each other or are short
it can be far off, so it is no general substitute for `sommerfeld` (see
[Model the inductive coupling of cables and leads](inductive-coupling.md)).

## 6. Parallel sweeps

Independent solves of a parameter study run in parallel with
`gf.sweep(..., n_jobs=...)`; see
[Run sweeps and Monte Carlo studies](sweeps-and-monte-carlo.md).

## 7. Memory

The image-family kernels never build per-pair difference tensors, switch
to row blocks for very large matrices, and evaluate `result.potential`
on large grids in blocks, so memory grows with $N^2$ for the reaction
matrix itself and not with the number of field points. If memory is the
limit, reduce $N$ (segment length, fewer electrodes per solve) first.

??? note "Changes since 0.15.0"

    - `sommerfeld_assembly="tabulated"` is the default since 2026.11; up
      to 0.15.0 the default was the per-pair assembly `"exact"`.
    - `solve_mutual_matrix` takes the Galerkin reaction as its
      off-diagonal since 2026.11; `offdiag="probe"`, which samples the
      potential at probe points, was the definition up to 0.15.0.
    - The default kernel of the image family is the exact line source
      since 2026.11 (ADR-0015); `kernel="point"` reproduces the numbers
      of 0.15.0.

## See also

- [Performance and scaling](../explanation/performance.md): measured run
  times and memory of this release.
- Tutorial [T09](../tutorials/T09_choosing_an_engine_and_convergence.ipynb):
  the convergence study behind step 2.
- [Run sweeps and Monte Carlo studies](sweeps-and-monte-carlo.md): parallel
  sweeps.
- [Diagnostics API](../reference/api/diagnostics.md): `HARD_LIMIT`,
  `SOFT_LIMIT`, `check_segment_budget`, `ResourceLimitError`.
