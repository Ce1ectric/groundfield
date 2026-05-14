# Engine theory — overview

`groundfield` ships eight numerical backends for the same physical
problem: the quasi-static potential field of a current-injected
electrode arrangement embedded in a horizontally stratified, semi-
infinite half-space. Every backend solves the same Sommerfeld
integral

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\int_0^{\infty} \bigl[
  e^{-\lambda |z - z_s|}
+ \Gamma_1(\lambda)\, e^{-\lambda (z + z_s)}
\bigr]\, J_0(\lambda s)\, d\lambda,
$$

with the recursive upward-looking reflection coefficient
$\Gamma_1(\lambda)$ built bottom-up from the per-interface Fresnel
coefficients $K_i = (\rho_{i+1} - \rho_i) / (\rho_{i+1} + \rho_i)$.
What differentiates the engines is **how** they evaluate this
integral.

This section gathers the mathematical and physical foundations for
each backend in a long-form, production-grade treatment. The API
reference below it (under [Solver](../api/solver.md)) carries the
auto-generated mkdocstrings output and stays close to the code; this
section instead derives, justifies, and bounds the methods.

## Family map

The eight backends fall into three methodologically distinct
families:

| Family | Backends | Discretisation | Computational form |
|---|---|---|---|
| Closed-form image-charge | `image`, `image_2layer`, `image_nlayer`, `cim` | wire segments | sum of point sources at real / complex image positions |
| Integral equation | `mom`, `mom_sommerfeld`, `bem` | wire segments | dense linear system on segment level |
| Volume PDE | `fem` | (s, z) triangular mesh | sparse linear system on mesh nodes |

The families share the same data model (`World`, `Electrode`,
`Conductor`, `Source`) and the same cluster-equipotentiality and
sum-of-currents constraints. They differ in the kernel, the test
function, and the discretisation domain.

## Decision tree

The recommended primary engine for a given problem follows the
soil-layer count and the desired confidence level:

```
n_layers = 1 (homogeneous)
   ├─ default              → image
   ├─ Sunde / Dwight check → mom
   └─ volume cross-check   → fem

n_layers = 2 (two-layer)
   ├─ default              → image_2layer       (or image_nlayer / image — auto-dispatches)
   ├─ independent kernel   → mom                (Galerkin on the same kernel)
   ├─ alternative weighting→ bem                (collocation, CIM kernel)
   └─ absolute reference   → mom_sommerfeld     (direct quadrature, slow)

n_layers ≥ 3 (multi-layer)
   ├─ primary              → cim                (complex images, closed form)
   ├─ alternative weighting→ bem                (collocation, same kernel)
   ├─ absolute reference   → mom_sommerfeld     (direct quadrature)
   └─ image_nlayer raises a clear ValueError —
     the real Stefanescu series is intentionally
     not implemented; ADR-0002 documents why.
```

`Engine.solve` automatically forwards `backend="image"` to
`image_2layer` for a `TwoLayerSoil` and to `image_nlayer` for a
`MultiLayerSoil`, so notebooks written for the homogeneous case keep
working when only the soil model is swapped.

## Cross-validation envelope

Cross-engine consistency is encoded in
`tests/test_cross_engines_extended.py`. Tolerances per pair:

| Pair | Tolerance | Reason for bound |
|---|---|---|
| any closed-form image vs. another closed-form image | $10^{-9}$ | exact reduction (e.g. `image_nlayer` → `image_2layer`) |
| closed-form image vs. `cim` | $5\,\%$ | matrix-pencil fit accuracy at low $P$ |
| closed-form image vs. `mom` / `bem` | $5\,\%$ | uniform-current vs. Galerkin / collocation |
| `mom_sommerfeld` vs. closed-form layered engines | $5\,\%$ | absolute reference; quadrature is the truth |
| `fem` vs. integral engines | $10\,\%$ | equivalent-hemisphere reduction |

Layer-contrast monotonicity (sweeping $\rho_2$ at fixed $\rho_1$
must produce a monotonically increasing cluster impedance) is a
basic physics consistency check enforced for every engine.

## Reading order

If you are coming to this section fresh, the suggested reading
order is

1. [`image`](image.md) — establishes the homogeneous baseline and
   the segment discretisation that all integral engines re-use.
2. [`image_2layer`](image_2layer.md) — first layered engine; the
   Tagg / Sunde series is the conceptual blueprint for the rest of
   the family.
3. [`mom`](mom.md) — Galerkin resolution scheme on the same
   kernels.
4. [`cim`](cim.md) — closed-form layered Green's function via
   complex images; the bridge from the literature on the
   matrix-pencil method to the engine family.
5. [`mom_sommerfeld`](mom_sommerfeld.md) — the absolute reference
   engine; useful for checking the closed-form approximations.
6. [`bem`](bem.md) — collocation alternative to the Galerkin MoM.
7. [`fem`](fem.md) — the only volume-PDE engine; provides an
   independent cross-check.
8. [`image_nlayer`](image_nlayer.md) — the dispatcher that ties the
   image-charge family together; reading it last makes its
   delegation rules transparent.

Each page contains:

- **Physical context** — what real-world problem this engine
  addresses, and the assumptions it inherits from `groundfield`.
- **Governing equations** — the differential / integral / spectral
  form the engine actually solves.
- **Numerical strategy** — how the equations are discretised,
  truncated, and stabilised.
- **Validity envelope** — geometry, frequency, contrast and mesh
  ranges where the engine is reliable.
- **Convergence and cost** — per-segment and per-mesh-node scaling.
- **Cross-validation notes** — which other engines should agree
  with it, and within which tolerances.
- **References** — the literature the engine is derived from.

## Architecture decisions

The selection heuristic and its alternatives are kept in two
architecture decision records:

- [ADR-0001](../adr/0001-two-layer-method.md) — the original
  argument for the two-engine setup (`image_2layer` + `mom`) for the
  typical study.
- [ADR-0002](../adr/0002-engine-family.md) — the extension to eight
  backends, the cross-validation envelope above, and the rationale
  for *not* implementing the real Stefanescu series for $n \ge 3$.
