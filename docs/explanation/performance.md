# Performance and scaling

This page tells you what a `groundfield` solve costs in time and memory,
which part of the computation the cost comes from, and which settings to
change first. Read it before a large study — a whole TN network, a sweep
of hundreds of cells, a three-dimensional `fem` solve — or when a solve is
slower than expected. Recipes for running large studies (parallel sweeps,
Monte Carlo, speeding up large networks) are in the
[how-to guides](../how-to/index.md).

!!! abstract "At a glance: measured costs"

    | Task | Size | Time, memory |
    |---|---|---|
    | galvanic TN network, 30 houses, homogeneous, point kernel | $N$ = 3687 | 0.26 s, 115 MiB |
    | the same, 80 houses | $N$ = 9712 | 1.98 s, 773 MiB |
    | the line kernel (default) on such networks | | 1.1 … 1.9 times the point kernel |
    | 30 houses, two-layer soil, 11 frequencies, point kernel | $N \approx$ 3 700 | 2.6 … 9.8 s |
    | inductive TN network, 30 houses, Sommerfeld earth return, 4 frequencies | 338 inductive branches | 12.2 s |
    | `fem`, axisymmetric model (ring, rod, ring with rod) | 7 341 … 223 112 nodes | 0.20 … 2.0 s |
    | `fem`, 3-D model (AP1 station) | 1.2 … 1.4 million nodes | 75 … 140 s, 4.3 GB |

    Measured as described [below](#measured-in-this-release); the `fem`
    numbers are those of the [fem page](engines/fem.md).

## Where the cost is

A solve of the image family has three parts:

1. **Reaction matrix.** For $N$ segments the kernel is evaluated for every
   pair: $O(N^2)$ time and, if the matrix is kept, $8N^2$ bytes. In
   two-layer soil every pair carries the Tagg / Sunde series, so the cost
   grows with the number of image terms, which depends only on the
   contrast $\lvert K\rvert$ (20 terms at $\lvert K\rvert = 0.5$, 277 at
   $\lvert K\rvert = 0.94$ for the default tolerance; table on the
   [`image_2layer`](engines/image_2layer.md#validation) page). An electrode
   through the layer interface switches every pair to the spectral kernel
   of ADR-0007, which is slower and needs more memory per depth group.
2. **Network solve.** The unknowns are one current per owner (each
   straight wire, and its part in each soil layer for a wire through the
   interface), one potential per node and one current per finite branch —
   a small dense system. A purely galvanic world is solved once and reused
   for every frequency; with inductive conductors one complex solve per
   frequency follows, which is cheap against the assembly.
3. **Earth-return assembly** of inductive conductors: one integral per
   pair of conductor sub-segments, the dominant cost of inductive worlds
   with the default `sommerfeld` model.

The `mom` family solves for one current per segment: its dense LU is
$O(N^3)$ and dominates for large $N$. `mom_sommerfeld` adds one
quadrature per distinct pair argument.

`fem` meshes the soil instead. Its cost is set by the node count of the
finer of its two mesh levels, not by $N$: about 4 GB per million nodes for
the axisymmetric model and 2.9 GB per million nodes for the 3-D model,
whose conjugate gradients run once per port and level. Both refuse a mesh
above `FemOptions.max_nodes` / `max_nodes_3d` (2 million nodes) before it
is built (`ResourceLimitError`); the [fem page](engines/fem.md) has the
measurements.

### Memory model

The kernels never build `(N, N, 3)` difference tensors. The reaction
matrix is materialised once per solve while it fits a budget of 1 GiB —
$8N^2$ bytes, i.e. up to $N$ = 11 585 segments in homogeneous soil,
$16N^2$ and $N$ = 8 192 for the two-layer series — and applied with the
same BLAS calls as before, which keeps results bit-identical. Beyond the
budget the kernels work in row blocks of 4 MiB with
$O(N\cdot\text{block})$ memory; results then differ at round-off (about
$10^{-15}$). Field evaluation (`FieldResult.potential`) runs in pieces of
at most 16 MiB, so a 200 × 200 grid of field points costs memory
proportional to the grid, not to grid times sources.

The resource limit of the engine, `Engine.max_segments` (default
`diagnostics.HARD_LIMIT`, 20 000 segments or nodal unknowns), stops a
world before anything of size $N$ is allocated (`ResourceLimitError`);
raise it per engine to solve a larger world on purpose. `SOFT_LIMIT`
(5 000) marks the size where solves take minutes rather than seconds on a
typical laptop; every `Engine.solve` at or above it emits a `UserWarning`
that says so (`solve_mutual_matrix` and `solve_mutual_field` check the
limit but emit no notice; `fem` emits it from the count of its display
segments, which is not its problem size — that is its mesh). The size of
that notice belongs to the engine as well, `Engine.warn_segments`
(default `SOFT_LIMIT`, read at solve time), so a workstation can move it
up together with `max_segments`.

??? note "Measurements and audit trail"

    Up to 0.15.0 the difference tensors needed 96 bytes per segment pair,
    1.25 GB for a 30-house network and a `MemoryError` at 80 houses
    (finding 10.56). The resource limit is finding 10.06, the notice's
    entry points are audit D2A-09, and `Engine.warn_segments` is user
    decision Q31.

## What to change first

In order of effect on a typical study:

1. **Earth-return model of inductive conductors.** The Sommerfeld
   assembly dominates inductive studies; keep the default
   `Engine.sommerfeld_assembly="tabulated"`, the faster and, by the
   measurements below, more accurate choice (`"exact"` integrates every
   segment pair and costs 17–25 times as much on the TN networks below).
   Leave conductors without an `inductance_model` where their inductance
   is not the question — then no assembly runs at all.
2. **Segment length.** Cost and memory grow with $N^2$. With the line
   kernel straight wires are exact at any segment length, so `ds` only
   needs to resolve the geometry (corners, rings, the distance between
   neighbouring wires and to a layer interface); a convergence study
   (`convergence_study`) shows when it does.
3. **Parallel cells** of a sweep or Monte Carlo study (`sweep(n_jobs=...)`).
4. **numba** (extra `fast`) for the line kernel's far field and the
   two-layer series; for catalogue sweeps in high-contrast two-layer soil
   `Engine.image_far_field = "point"` (point-form far images, 3e-6 …
   4.3e-4 on networks).
5. **Contrast and electrodes through the interface** set the cost of
   two-layer worlds; the series length follows from $\lvert K\rvert$
   alone.
6. **`fem`** only where it is needed as a reference: the axisymmetric model
   takes seconds, the 3-D model minutes and gigabytes per solve.

## Measured in this release

How the numbers were measured: synthetic AP1-like networks of the
review's performance study (R8 workloads) — a substation with a 4 × 4 m
ring and three rods, 10 to 80 houses with their foundations on both sides
of four streets, PEN cables with service drops, 100 Ω·m or
500/100 Ω·m with $h_1$ = 5 m, ds = 0.5 m — built with the layout builder.
The release work-streams ran v0.15.0 and the release candidate back to
back in separate processes, one BLAS thread, on a shared machine with two
cores; times are wall-clock (best of two for the short ones), memory is
the `tracemalloc` peak, which is deterministic. Wall-clock ratios vary
with the machine load; memory ratios do not. The scripts belong to the
release evidence; the memory gates are part of the test suite
(`tests/test_kernel_performance.py`), the timing gates run with
`--run-benchmark`.

### Galvanic solves (point kernel, bit-identical to v0.15.0)

| Workload | $N$ | v0.15.0 time / memory | 2026.11 time / memory |
|---|---:|---|---|
| W1: 30 houses, homogeneous, 50 Hz | 3687 | 2.7 s / 1247.6 MiB | 0.26 s / 115.0 MiB |
| W1-80: 80 houses, homogeneous | 9712 | `MemoryError` (8.4 GiB of tensors) | 1.98 s / 773.2 MiB |
| W2a: 30 houses, two-layer, rods inside layer 1, 11 frequencies | 3669 | 94.8 s / 1235.5 MiB | 9.8 s / 218.7 MiB |
| W2: 30 houses, two-layer, rods through $h_1$, 11 frequencies | 3690 | 4.2 s / 1249.7 MiB | 2.6 s / 510.2 MiB |
| W4: potential of W1 on a 50 × 50 grid | 3687 | 0.81 s / 844.1 MiB | 0.16 s / 15.5 MiB |
| W4-200: potential of W1 on a 200 × 200 grid | 3687 | — | 1.6 s / 17.0 MiB |
| $Z_G$ (`solve_mutual_matrix`), 20 houses, two-layer | 2466 | 82.7 s / 604.1 MiB | 2.65 s / 146.0 MiB |

??? note "Measurements and audit trail"

    The gains come from component-wise row-block kernels instead of
    tensors, one reaction-matrix assembly per solve instead of two, the
    two-layer series summed into one matrix before the product (optionally
    compiled with numba, bit-identical) and the direct assembly of $Z_G$
    instead of applying the kernel to an identity matrix ($O(N^3)$ per
    image term; findings 10.56–10.59, 10.62). The W2 peak was the grouped
    spectral correction of the cross-layer path; since wave 3 of the
    2026.11 work a large depth group is planned once from the range of its
    distances and evaluated in row chunks (bit-identical), and beyond the
    budget the path works in row blocks like the series path (see
    *Electrodes through the layer interface* below).

### The default line kernel

The exact line kernel ([kernels](kernels.md), ADR-0015) adds the exact
near-pair integrals and the quadrupole far field over all pairs. Its cost
relative to the point kernel, measured on the release state:

| Workload | $N$ | point: time / memory | line: time / memory | ratio (time) | $Z_\text{in}$, line vs point |
|---|---:|---|---|---:|---:|
| 16 house grids, homogeneous | 1920 | 0.043 s / 38.8 MiB | 0.165 s / 47.7 MiB | 3.8 (5.8 without numba) | — |
| 16 house grids, 500 / 100 Ω·m, $h_1$ = 5 m | 1920 | 0.332 s / 64.4 MiB | 0.519 s / 69.7 MiB | 1.6 (1.4 without numba) | — |
| W1: 30 houses, homogeneous | 3668 | 0.83 s / 113.4 MiB | 1.55 s / 155.7 MiB | 1.9 | +0.35 % |
| W1-80: 80 houses, homogeneous | 9668 | 6.19 s / 758.3 MiB | 6.94 s / 856.2 MiB | 1.1 | +0.27 % |

Install the extra `fast` (numba) for the compiled far field;
`kernel="point"` reproduces the old numbers and cost.

??? note "Measurements and audit trail"

    The 16-house rows are the release state (best of three rounds of fresh
    processes, interleaved, with other agents on the machine, load
    6 … 13); the homogeneous one is the case documented with
    `Engine.kernel`. The W1 rows were measured under a machine load of
    about 5.6 on two cores (numba installed), so their absolute times are
    about three times those of the galvanic table above; the ratio and the
    memory are the meaningful columns. The networks differ slightly from
    the v0.15.0 table ($N$ = 3668 instead of 3687) because the layout
    builder now joins PEN cables at junction nodes instead of 5 cm rods.

### Far images in two-layer soil

In two-layer soil every Tagg/Sunde image within reach carries the
far-field term of the line kernel. Its cost follows the image density:
high contrasts and thin top layers put hundreds of images within reach,
and with long pieces below a thin top layer the Galerkin integrals of their
near image pairs add to it (each long piece is within 7.5 segment lengths
of every short segment's image; the near-pair criterion of the kernel).
`Engine.image_far_field = "point"` keeps far images as point sources and
leaves out their far-field term; the near image pairs stay Galerkin
integrals on both paths, so on the cross-layer path it takes 0.49 … 0.94
of the default's time. Headline numbers (line kernel, ds = 0.5 m, 50 Hz,
numba installed; † the cross-layer path):

| world | $N$ | default | `"point"` |
|---|---:|---:|---:|
| 16 house grids, 500 / 100 Ω·m, $h_1$ = 5 m | 1920 | 0.54 s | 0.48 s |
| AP1 station, 40 % penetration † | 2711 | 1.35 s | 0.96 s |
| TN network, 40 houses, 1000 / 30 Ω·m, $h_1$ = 1.5 m † | 4877 | 4.03 s | 1.99 s |
| the same without numba † | 4877 | 9.46 s | 2.33 s |

`"point"` moves networks by 3e-6 … 4.3e-4 against the default: a choice
for catalogue sweeps, not for fields near the interface.

??? note "Measurements and audit trail"

    Audit W3A measured that the cost of the far-image term followed the
    longest segment of the world and the image density; since then an
    image is evaluated on the pairs whose longer segment reaches it, the
    reaction matrix's part on its upper half, with one division per
    evaluation. `Engine.image_far_field = "point"` is the author's decision
    of 2026-11-15; its near image pairs on both paths are audit W3R2A-01.

    Line kernel, ds = 0.5 m, 50 Hz, best of three solves, one fresh process
    per row, the four trees of a row run one after the other (other agents
    on the machine, load 10 … 18; the times of stream fw3xl measured before
    its last commit, which changed the summation of the sparse image
    corrections and nothing else; peaks of the final code). The rows marked
    † solve on the cross-layer path (wires at or below $h_1$; the TN network
    in 1000 / 30 Ω·m through its transformer rods); their "before wave 3",
    "now" and `"point"` columns were measured again after audit W3R2A-01
    (stream fw3r2a, load 4 … 9), the "wave 3" column is that of stream
    fw3xl:

    | world | $N$ | numba | before wave 3 | wave 3 | now | `"point"` | peak now / `"point"` |
    |---|---:|---|---:|---:|---:|---:|---|
    | 16 house grids, 100 Ω·m | 1920 | yes | 0.14 s | 0.17 s | 0.18 s | 0.18 s | 47.7 / 47.7 MiB |
    | 16 house grids, 500 / 100 Ω·m, $h_1$ = 5 m | 1920 | yes | 0.43 s | 0.71 s | 0.54 s | 0.48 s | 69.7 / 66.4 MiB |
    | the same with 8 pieces of 9.4 m | 1928 | yes | 0.69 s | 1.97 s | 1.03 s | 0.92 s | 94.7 / 94.6 MiB |
    | 16 house grids below $h_1$ = 0.5 m † | 1920 | yes | 0.27 s | 0.91 s | 0.54 s | 0.40 s | 80.5 / 80.5 MiB |
    | the same with 8 pieces of 9.4 m † | 1928 | yes | 0.32 s | 2.17 s | 1.66 s | 1.44 s | 158.1 / 158.1 MiB |
    | 16 house grids, 30 / 1000 Ω·m, $h_1$ = 0.1 m † | 1920 | yes | 0.26 s | 2.42 s | 1.12 s | 0.64 s | 87.0 / 86.9 MiB |
    | 16 house grids, 1000 / 30 Ω·m, $h_1$ = 0.1 m † | 1920 | yes | 0.27 s | 2.39 s | 1.13 s | 0.64 s | 87.0 / 86.9 MiB |
    | AP1 station, 40 % penetration † | 2711 | yes | 1.29 s | 1.86 s | 1.35 s | 0.96 s | 137.1 / 136.5 MiB |
    | TN network, 40 houses, 500 / 100 Ω·m, $h_1$ = 5 m | 4877 | yes | 2.59 s | 4.38 s | 3.24 s | 2.71 s | 407.5 / 407.5 MiB |
    | TN network, 40 houses, 1000 / 30 Ω·m, $h_1$ = 1.5 m † | 4877 | yes | 2.19 s | 10.06 s | 4.03 s | 1.99 s | 404.1 / 404.1 MiB |
    | 16 house grids, 500 / 100 Ω·m, $h_1$ = 5 m | 1920 | no | 1.47 s | 2.56 s | 1.83 s | 1.50 s | 77.0 / 77.0 MiB |
    | 16 house grids below $h_1$ = 0.5 m † | 1920 | no | 0.34 s | 1.89 s | 0.85 s | 0.46 s | 80.5 / 80.5 MiB |
    | AP1 station, 40 % penetration † | 2711 | no | 1.46 s | 3.99 s | 2.17 s | 1.06 s | 140.5 / 136.4 MiB |
    | TN network, 40 houses, 1000 / 30 Ω·m, $h_1$ = 1.5 m † | 4877 | no | 2.57 s | 27.06 s | 9.46 s | 2.33 s | 404.0 / 404.0 MiB |

    Measured with the trees in alternation (three rounds), the 40-house
    network in 1000 / 30 Ω·m takes 1.84 times the solve before wave 3 with
    numba and 3.63 times without (stream fw3xl, load 6 … 13; 1.84 and 3.68
    in stream fw3r2a's rounds, load 4 … 9); `"point"` 0.91 and 0.90 times
    (0.83 and 0.85 with the line form of the interface image only, before
    audit W3R2A-01). Before audit W3R2A-01 `"point"` kept only the
    interface image on the cross-layer path, which was faster below thin
    top layers (1.2 times the solve before wave 3 instead of 1.5 … 2.4,
    with long pieces 1.2 instead of 4.3 … 4.5) and lost up to 58 % on long
    segments below the interface.

### Electrodes through the layer interface

The ADR-0007 path of `image_2layer` (wires at or below $h_1$) costs one
spectral evaluation per pair of segment depths; large depth groups are
planned once and evaluated in row chunks. Its peak memory is that of the
series path: 16 house grids entirely below $h_1$ = 0.5 m need 21.9 bytes
per segment pair (the series path 18.3), and beyond the matrix budget the
path works in row blocks with memory linear in $N$ (36 house grids,
$N$ = 4320: 34.8 MiB in block mode). On the AP1 station of the roadmap
with 40 % penetration ($N$ = 2711) it takes 1.35 s with the line kernel
(2.17 s without numba) and 0.96 s with `image_far_field="point"`; the
same world with 4 m rods that stay in the top layer solves on the series
path in 1.27 s.

??? note "Measurements and audit trail"

    The AP1 station of the roadmap — a 4 × 4 m ring at 0.6 m and three
    6 m rods from 0.6 m through $h_1$ = 5 m (500 / 100 Ω·m), 60 houses with
    foundations (10 × 10 m, 2 × 2 meshes at 0.8 m) at 5 … 40 % penetration,
    PEN cables, ds = 0.5 m, 50 Hz — before and after wave 3 of the 2026.11
    work (best of three, numba installed, a shared machine; memory:
    `tracemalloc` peak):

    | penetration | $N$ | line kernel before → after | point kernel before → after | peak before → after | the same world with 4 m rods (series path) |
    |---:|---:|---|---|---|---|
    | 5 % | 311 | 0.82 → 0.43 s | — | 19.4 → 20.8 MiB | 0.06 s, 8.5 MiB |
    | 10 % | 671 | 0.84 → 0.50 s | 0.85 → 0.42 s | 27.8 → 28.5 MiB | 0.11 s, 18.7 MiB |
    | 20 % | 1031 | 0.89 → 0.62 s | — | 42.5 → 39.6 MiB | 0.22 s, 25.0 MiB |
    | 40 % | 2711 | 1.28 → 1.65 s | 1.14 → 0.70 s | 278.3 → 136.6 MiB | 1.27 s, 127.1 MiB |

    The spectral amplitudes are eliminated in closed form (one LAPACK solve
    per quadrature node before, half the time of the path): the point
    kernel, which sees only that change, is 1.6 … 2.0 times faster. The
    line kernel carries the layered correction in line form now (a rod
    through the interface no longer depends on the segment length); since
    audit W3A its far-field terms cost about what the closed-form
    amplitudes saved (1.35 s at 40 % penetration, 1.29 s before wave 3;
    2.17 s without numba). Before wave 3, 16 house grids entirely below
    $h_1$ = 0.5 m needed 41.2 bytes per segment pair, and the 36 house
    grids 1008 MiB without row blocks (finding 10.56).

### Inductive conductors

| Workload | v0.15.0 | 2026.11, `sommerfeld_assembly="exact"` | 2026.11, `"tabulated"` (default) |
|---|---|---|---|
| W3: 2 houses, Neumann PEN, Sommerfeld earth return, 11 f | 35.3 s | 1.9–2.1 s | 1.2–1.5 s |
| W3: 30 houses, same | ≈ 3600 s (estimate) | 242 s | 7.8–9.1 s |
| W3c: 30 houses, `carson_series` | 16.4 s | 3.3 s | — |
| TN generator, 8 houses, Neumann PEN (78 inductive branches), 4 f | — | 16.6 s | 1.0 s |
| TN generator, 30 houses, same (338 inductive branches) | — | 306 s | 12.2 s |

The tabulated assembly shares one table of the spectral integral per pair
of depth groups and frequency, and agrees with a converged reference
better than the per-pair rule (substation impedance $5\cdot 10^{-10}$
against $1.1\cdot 10^{-8}$ on W3; input impedance of the 8-house TN
network $2\cdot 10^{-10}$ against $1.2\cdot 10^{-8}$). `carson_series` is
an approximation for long parallel lines, not a replacement of
`sommerfeld` ([earth return](earth-return.md)), and no faster than
necessary to be a check.

??? note "Measurements and audit trail"

    CPU times of the W3 rows measured by the coupling work-stream, of the
    TN rows by the stream that made the tabulated assembly the default
    (user decision Q16; `TnNetworkGenerator`, homogeneous 100 Ω·m,
    ds = 1 m, 50/250/550/1000 Hz). The per-pair path gains from one
    multi-frequency assembly per solve (geometry, λ grid and Bessel tables
    once per pair; bit-identical) and the resolved λ grid (finding 10.60).
    `carson_series` moved by its own fix (F32).

### Other measurements

- **`mom_sommerfeld`, three layers.** The depth-group tabulation of the
  reflected kernel (default) took a 40 m grid in 1000/30/200 Ω·m soil at
  ds = 1.25 m from 66.6 s to 0.30 s and a village of eleven rings
  ($N$ = 273) from 34.1 s to 1.25 s, within $10^{-6}$ of the quadrature
  (benchmark B12).
- **`sweep(n_jobs=...)`.** Parameter cells run in a process pool with one
  BLAS thread per worker; the result is bit-identical to `n_jobs=1` when
  the parent also runs a single BLAS thread. The review measured 1.98
  times the throughput on two workers. The LU factorisation is 0.2–1.5 %
  of a solve, so BLAS threads are not the lever; parallel cells are.
- **Import.** `import groundfield` loads its heavy dependencies lazily:
  223 ms instead of 798 ms, 346 instead of 1169 modules.
- **`fem`.** The axisymmetric model takes 0.20 s, 0.58 s and 2.0 s for a
  ring, a 3 m rod and a ring with a central rod in two layers (7 341 /
  28 934, 18 868 / 74 871 and 56 079 / 223 112 nodes at levels 0 / 1).
  The 3-D model takes 26 s and 1.8 GB for a ring of 12.6 m (600 000 nodes
  at the finer level) and 75 … 140 s and 4.3 GB for the AP1 station
  (28 m of wire, 1.2 … 1.4 million nodes); `FemOptions.max_nodes_3d`
  (2 million nodes) refuses meshes of more than about 6 GB before they are
  built. Details and conditions on the [fem page](engines/fem.md).

??? note "Measurements and audit trail"

    The tabulation of `mom_sommerfeld` is finding 10.61, the parallel
    sweep finding 10.64, the lazy import finding 10.72.

## Literature

- Review pass 10 (2026-11-07), Appendix C (performance baseline of
  v0.15.0) and findings 10.56–10.65.
- [ADR-0010](../adr/0010-tier0-performance.md) (tier-0 performance work of
  earlier releases) and [ADR-0014](../adr/0014-backend-registry-and-pipeline.md)
  (registry and shared pipeline).
