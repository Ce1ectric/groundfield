# Explanation

This section explains what `groundfield` computes, with which equations,
under which assumptions and how well. It is written to be cited: every
engine page states its mathematical and physical context, the equations
and discretisation actually implemented, its validity range and the
literature, and the accuracy figures are generated from the benchmark
suite rather than written by hand.

| Page | Content |
|---|---|
| [Concepts](concepts.md) | position in the software family, the objects of a computation, modelling assumptions, the reduced $\rho$–$f$ model |
| [Engine family](engines/index.md) | the governing equations, the two current models, the registered backends, how to choose, deprecated names |
| [`image`](engines/image.md) · [`image_2layer`](engines/image_2layer.md) · [`mom`](engines/mom.md) · [`mom_sommerfeld`](engines/mom_sommerfeld.md) · [`fem`](engines/fem.md) | one theory page per engine |
| [Kernels](kernels.md) | point source and exact line source, the Galerkin self term, field points near wires |
| [Earth return](earth-return.md) | inductive coupling of conductors: perfect mirror, Carson, Sommerfeld / Pollaczek, skin effect |
| [Concrete shells](concrete-shells.md) | the cylindrical shell of foundation electrodes and its single mechanism |
| [Sources and measurement](sources-and-measurement.md) | two-terminal sources, junctions, input / cluster / electrode impedance, the fall-of-potential loop |
| [Validation and benchmarks](validation.md) | generated tolerance table per backend, comparison figures, kernel and field checks |
| [Performance and scaling](performance.md) | memory model, measured run times of this release, how they were measured |

The architecture decision records (ADRs) document why the code is built
the way it is; each carries dated amendments where later releases changed
a decision. Release 2026.11 amended ADR-0002, 0004, 0005, 0006, 0007,
0008, 0009 and 0012 and added
[ADR-0013](../adr/0013-calendar-versioning.md) (calendar versioning),
[ADR-0014](../adr/0014-backend-registry-and-pipeline.md) (backend registry
and shared solve pipeline),
[ADR-0015](../adr/0015-line-source-kernel-default.md) (exact line-source
kernel as default) and
[ADR-0016](../adr/0016-conforming-fem.md) (conforming finite elements in
`fem`).
