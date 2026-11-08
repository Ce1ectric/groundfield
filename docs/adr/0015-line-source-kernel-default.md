# ADR-0015: Exact line-source kernel as the default of the image family

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-11-07 |
| **Deciders** | Project maintainers |
| **Scope** | `Engine.kernel`; the reaction matrix and the field evaluation of `image`, `image_2layer` (and the deprecated names that run them), `solve_mutual_matrix`, `solve_mutual_field`, `FieldResult.potential`; `PointSource.direction` |

## Context

Up to 0.15.0 the image family placed the current of every wire segment at
the segment midpoint (point kernel) and used the collocation self term
$2\ln(L/a)/L$ on the diagonal. Review pass 10 (finding 10.28) showed that
this kernel converges to a **biased** limit:

- Per row of the reaction matrix the self term is $+0.61/L$ too large and
  the two collinear neighbours $-0.77/L$ too small; the row sum errs by
  about $-0.23/L$ for every segment length $L$. Because the row sum itself
  scales like $1/L$, the relative error of the resistance does not vanish
  under refinement: rods, rings and strips converge 0.9–1.9 % below the
  exact solution of the uniform-current model. A convergence study cannot
  see it.
- Field points near a wire see point sources: the surface potential
  spikes at every segment midpoint and is several per cent off above a
  wire at typical segment lengths (finding 10.33).
- Near a layer interface the point form of the interface image needs
  segments much shorter than the distance to the interface (−46 % for a
  ring 5 cm below $h_1$ at ds = 0.5 m).
- The kernel error has the opposite sign of the uniform-current ansatz
  error, so for impedances the two partly cancel — which made the old
  numbers look closer to some references than they were.

The review proposed the exact line-source kernel as an opt-in with a
default decision at 1.0. The user decided to make it the default in this
release, with `kernel="point"` reproducing the old numbers.

## Decision

1. **`Engine.kernel`** takes `None` (default), `"line"` or `"point"`.
   `None` lets the backend choose its preferred kernel: `"line"` for the
   image family, the backend's own kernel (`"point"`) for every other
   backend. An explicit `"line"` on a backend without it raises
   `BackendCapabilityError`. The request is carried in the value, so an
   engine rebuilt from JSON or from a result's provenance block runs like
   the original; `metadata["kernel"]` records the kernel that ran.
2. **Galerkin form.** The line kernel is the segment-averaged potential of
   uniform line currents with the reduced kernel
   $1/\sqrt{r^2 + a^2}$ — the Galerkin pairing dual to the uniform-current
   ansatz, symmetric, and exact for a straight wire at any segment length.
   A Galerkin self term with *collocated* exact mutual terms was rejected:
   it biases every interior row by about $-0.6/L$, worse than the point
   kernel.
3. **Evaluation.** Pairs within six segment lengths are integrated
   exactly (closed form for parallel pairs, graded Gauss–Legendre with the
   exact inner line potential otherwise); farther pairs use the point form
   plus the quadrupole term of the segment average, remainder
   $O((L/R)^4)$. The Tagg/Sunde images use the point form unless closer
   than six segment lengths. On the cross-layer path pairs in different
   layers carry the transmitted resistivity $2\rho_1\rho_2/(\rho_1+\rho_2)$
   and the interface image is a Galerkin integral for near pairs, which
   keeps the matrix reciprocal.
4. **Field evaluation.** Line-kernel results store the unit tangent of
   every source segment (`PointSource.direction`), and
   `FieldResult.potential` evaluates exact line potentials near the
   segments. Results without directions (written by 0.15.0) are evaluated
   with the point kernel.
5. **The `mom` family keeps the point kernel.** Its point-matching scheme
   is defined with point sources; cross-checks between `image` and `mom`
   pin `kernel="point"` so that they compare current models on one kernel.
6. **Cost control.** The far-field term over all pairs is the expensive
   part. It is compiled with numba when available (optional extra `fast`,
   bit-identical blocks) and runs in NumPy otherwise; `kernel="point"`
   remains the opt-out.

## Consequences

**Numbers change.** Grounding resistances of single electrodes rise by
about 1–2 % at typical segment lengths (the point kernel's bias); network
quantities move less — the input impedance of a synthetic 30-house network
by +0.35 % (performance page). The generated kernel check of the
validation page shows the size per geometry and segment length; the
release notes list the moved golden cases. What remains against the true
thin-wire solution is the uniform-current ansatz itself (about +0.9 % for
the 3 m rod of benchmark B2, larger near grid corners for field
quantities, see the `image` engine page).

**Cost.** The `Engine.kernel` documentation gives the measured factor for
a 16-house solve ($N$ = 1920): 2.7 with numba, 5.7 without; the
performance page puts it into the context of whole studies.

**Accuracy statements become clean.** With the kernel error removed, the
remaining deviations of the image family are attributable to the current
model, which `mom` checks — the separation the validation page relies on.

## Alternatives considered

- **Keep the point kernel as default, line opt-in** (the review's
  proposal). Rejected: every study would carry a known, invisible
  1–2 % bias by default.
- **Far-field cut-off** (no quadrupole term beyond some distance): about
  1.5 times the point-kernel cost at about $10^{-4}$ accuracy instead of a
  few $10^{-6}$. Not taken; the numba path makes the full form affordable.
- **Quadrupole terms for every Tagg/Sunde image**: at most 0.05 % gain on a
  straight strip in two-layer soil for about twice the series assembly
  cost. Not taken.

## References

- Review pass 10 (2026-11-07), findings 10.28 and 10.33, decision
  question 4.
- Harrington, R. F. (1968). *Field Computation by Moment Methods*.
  Macmillan, New York.
- Wilton, D. R., et al. (1984). Potential integrals for uniform and linear
  source distributions on polygonal and polyhedral domains. *IEEE
  Transactions on Antennas and Propagation* AP-32(3), 276–281.
- The [kernels page](../explanation/kernels.md) of the documentation.
