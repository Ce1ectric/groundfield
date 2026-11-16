# Explanation

This section explains what `groundfield` computes, with which equations,
under which assumptions and how well. Read it when you want to know why a
result looks the way it does, which engine to trust for a question, or
how to cite the method. Every engine page states its physical model, the
equations and discretisation actually implemented, its validity range and
the literature; the accuracy figures are generated from the benchmark
suite rather than written by hand.

**Start here:** [Concepts](concepts.md) for the overall picture, then
[Engines](engines/index.md) to choose the engine for your study and to see
where each one must not be used.

| Page | Content |
|---|---|
| [Concepts](concepts.md) | the objects of a computation, what happens in a solve, what a result contains, the modelling assumptions, the reduced $\rho$–$f$ model |
| [Engines](engines/index.md) | which engine for which question: decision path, accuracy, cost, limits of use; the governing equations and the two current models |
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
