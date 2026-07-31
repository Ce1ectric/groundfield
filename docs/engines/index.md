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

!!! warning "Eight backends, six distinct computations"

    Two entries in that map are **aliases in disguise**, and a
    cross-validation table must not count them twice:

    | Backend | Accepts | Actually computes |
    |---|---|---|
    | `cim` | $n \le 2$ | the `image` ($n = 1$) / `image_2layer` ($n = 2$) closed-form kernel, bit-identical; no complex-image fit runs |
    | `bem` | $n \le 2$ | the `mom` reaction matrix and constraint solve, identical to 4e-16 |

    Both raise `NotImplementedError` for $n \ge 3$ (since 0.11.0 —
    the shared complex-image kernel was structurally incomplete,
    audit 2026-07-08 WP-E), so **`mom_sommerfeld` is currently the
    only engine that solves an $n \ge 3$ soil with the layered
    Green's function**; `fem` adds an independent volume-PDE check
    with its documented equivalent-hemisphere bias.

    Each result records this: `metadata['reduces_to']` names the
    backend the numbers are identical to, and
    `metadata['cim_fit_used'] = False` states that no complex-image
    fit entered the answer.

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
   ├─ resolution scheme    → mom                (Galerkin on the same kernel)
   ├─ independent kernel   → mom_sommerfeld     (direct quadrature, slow — the real cross-check)
   ├─ independent form     → fem                (volume PDE, ≤ 10 % hemisphere bias)
   └─ cim / bem run, but reproduce image_2layer / mom
     bit-for-bit — no independent information.

n_layers ≥ 3 (multi-layer)
   ├─ primary              → mom_sommerfeld     (full layered Green's function, direct quadrature)
   ├─ independent form     → fem                (volume PDE, the only cross-check available here)
   ├─ cim / bem raise NotImplementedError —
   │  the historic complex-image kernel was
   │  structurally incomplete (audit 2026-07-08).
   └─ image_nlayer raises a clear ValueError —
     the real Stefanescu series is intentionally
     not implemented; ADR-0002 documents why.
```

For $n \ge 3$ there is therefore exactly **one** production path
(`mom_sommerfeld`) plus one independent sanity check (`fem`). If you
need speed in that regime, reduce the stack to an equivalent two-layer
soil first and state the reduction, rather than reaching for `cim`.

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
| closed-form image vs. `cim` | $10^{-9}$ | **not a check** — same kernel, same numbers (`cim` reduces to `image` / `image_2layer`) |
| `mom` vs. `bem` | $10^{-9}$ | **not a check** — identical reaction matrix (4e-16) |
| closed-form image vs. `mom` / `bem` | $5\,\%$ | uniform-current vs. Galerkin / collocation |
| `mom_sommerfeld` vs. closed-form layered engines | $5\,\%$ | absolute reference; quadrature is the truth |
| `fem` vs. integral engines | $10\,\%$ | equivalent-hemisphere reduction |

The two rows marked *not a check* are regression guards on an
identity, not evidence about the physics. Only the last two rows
compare methodologically distinct computations.

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
4. [`cim`](cim.md) — the complex-image theory and the spectral
   fit of $\Gamma_1(\lambda)$; read it for the method, not for a
   solver you would choose (it reduces to `image_2layer`).
5. [`mom_sommerfeld`](mom_sommerfeld.md) — the reference engine, and
   the only $n \ge 3$ path; useful for checking the closed-form
   approximations.
6. [`bem`](bem.md) — collocation alternative to the Galerkin MoM,
   numerically identical to `mom` on the soils it accepts.
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
  Read its **amendments** as well: they revoke `cim` as the primary
  $n \ge 3$ engine and revoke `cim` / `bem` as independent
  cross-checks.
