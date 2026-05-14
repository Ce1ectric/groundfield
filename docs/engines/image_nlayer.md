# `image_nlayer` — image-charge dispatcher for n-layer soil

## Physical context

The image-charge family of solvers represents the layered Green's
function as a sum of point sources at mirrored positions in the
soil. This representation is closed-form for two regimes:

- **n = 1 (homogeneous).** A single source plus its air-mirror at
  $z = -z_s$ reproduces the Neumann boundary at $z = 0$ exactly.
- **n = 2 (two-layer).** $\Gamma_1(\lambda) \equiv K_1$ is constant
  in $\lambda$, so the multiple-reflection expansion of
  $1 / (1 - K_1 e^{-2\lambda h_1})$ is a geometric series whose
  coefficients are the well-known $K_1^n$ Tagg / Sunde weights.

For $n \ge 3$ the recursive reflection coefficient

$$
\Gamma_1(\lambda) \;=\;
\frac{K_1 + \Gamma_2(\lambda)\, e^{-2\lambda h_2}}
     {1 + K_1\, \Gamma_2(\lambda)\, e^{-2\lambda h_2}}
$$

becomes a non-trivial rational function of $e^{-2\lambda h_i}$ that
no longer collapses to a clean geometric series in the spatial
domain. Stefanescu / Sunde (1968, ch. 3.5) describe a doubly-nested
expansion that recovers a real-image series, but its
implementation is fragile for hard contrasts: the convergence
factor at every interface is $|K_i|$, and the doubly-nested loop
has to converge in *every* nested level. For practical typical
contrasts this works only with non-trivial Aitken / Pade
acceleration on top.

`image_nlayer` is the **dispatcher** that ties the closed-form
image-charge family together. Rather than carrying a fragile
$n \ge 3$ implementation, it delegates to the engines that are
designed for that regime ([`cim`](cim.md), [`bem`](bem.md),
[`mom_sommerfeld`](mom_sommerfeld.md)) and raises a clear error
otherwise. ADR-0002 documents this design decision.

## Dispatch table

| `n_layers` | Effective backend | Notes |
|---|---|---|
| 1 | [`image`](image.md) | Homogeneous half-space; the soil is internally cast to `HomogeneousSoil` if a `MultiLayerSoil` with one layer is supplied. |
| 2 | [`image_2layer`](image_2layer.md) | Tagg / Sunde geometric series. A `MultiLayerSoil` with exactly two entries is internally cast to `TwoLayerSoil`. |
| ≥ 3 | — (raises `ValueError`) | The error message points to `cim`, `mom_sommerfeld`, or `bem` for the regime. |

The returned `FieldResult.backend` is rewritten to `"image_nlayer"`
so that cross-engine comparisons see one unified label per backend
selection. The original delegate is preserved in
`FieldResult.metadata['dispatched_to']`.

## Why no $n \ge 3$ implementation?

Three reasons, each independently sufficient:

1. **Convergence sensitivity.** The doubly-nested Stefanescu series
   converges as $\prod_i |K_i \Gamma_{i+1}|^{n_i}$. For typical
   parameters with $|K_i| \approx 0.5$–$0.9$, the series can take
   $\gtrsim 10^4$ terms in the worst-case nested combinations.
   Without Aitken / Pade acceleration the engine would routinely
   fail to converge inside `max_terms`.

2. **Methodological redundancy.** The complex-image method (`cim`)
   solves the same problem with $\sim 8$ complex exponentials and
   gives a closed-form spatial Green's function for *any* $n$. The
   per-evaluation cost is independent of the layer count, and the
   matrix-pencil fit is numerically robust for the typical contrast
   range.

3. **Cross-validation strength.** `mom_sommerfeld` provides the
   absolute reference for $n \ge 3$ (direct numerical Sommerfeld
   quadrature), and `bem` provides an alternative weighting (CIM
   kernel, collocation). Adding a fragile real-image series would
   not increase confidence — it would just add another engine that
   the cross-validation matrix would have to discount when it
   diverges.

The trade-off is documented in ADR-0002 and revisited every time the
supported use-case scope expands.

## Notebook ergonomics

`Engine.solve` automatically forwards `backend="image"` to:

- `image_2layer` if the world holds a `TwoLayerSoil`,
- `image_nlayer` if the world holds a `MultiLayerSoil`,

so notebooks written for the homogeneous case keep working when only
the soil model is swapped. The path through `image_nlayer` for an
$n \ge 3$ soil therefore terminates with a clear `ValueError`
suggesting the appropriate alternative engine — useful safety rail
for users who copy-paste a 2-layer setup and only later upgrade the
soil model.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, or `MultiLayerSoil` |
| Effective layer count | 1 or 2 only — `n ≥ 3` raises `ValueError` |
| Frequency | inherited from delegate (quasi-static, $f < 1\,\text{kHz}$) |
| Electrode placement | every segment must lie inside the upper layer |

## Cross-validation notes

By construction the dispatcher reproduces its delegate bit-exactly:

| Stack | Delegate | Expected agreement |
|---|---|---|
| `HomogeneousSoil` | `image` | $10^{-9}$ (identity) |
| `TwoLayerSoil` | `image_2layer` | $10^{-9}$ (identity) |
| `MultiLayerSoil`, $n = 1$ | `image` (auto-cast) | $10^{-9}$ |
| `MultiLayerSoil`, $n = 2$ | `image_2layer` (auto-cast) | $10^{-9}$ |
| `MultiLayerSoil`, $n \ge 3$ | — | raises |

The dispatcher therefore appears in the cross-engine test matrix
as a "free" extra engine for the $n = 1, 2$ cases, with a tolerance
of $10^{-9}$ against its delegate.

## References

The dispatcher itself does not introduce new mathematics. The
delegate engines carry their own references on
[`image`](image.md) and [`image_2layer`](image_2layer.md). For the
$n \ge 3$ design decision see:

- **Sunde, E. D.** (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, ch. 3.5. Stefanescu series for $n$ layers and
  why it converges slowly for high contrasts.
- **Colominas, I., París, J., Navarrina, F. & Casteleiro, M.**
  (2012). Improvement of computer methods for grounding analysis
  in layered soils by using high-efficient convergence acceleration
  techniques. *Adv. Eng. Soft.* 44 — Aitken / Pade
  reformulations that would be required to make a real n-layer
  image series practical.
- **ADR-0002** — the in-repo justification of the dispatcher
  design.

## Related material

- API reference: `groundfield.solver.image_nlayer`.
- ADR-0002 — engine selection heuristic.
- Notebook `04_image_nlayer.ipynb` — exercises the dispatch table
  including the deliberate `ValueError` for $n = 3$ stacks.
