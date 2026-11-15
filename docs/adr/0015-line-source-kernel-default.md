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
  cost. Not taken in 2026.11.0 at first; taken in wave 3 with a taper
  (amendment below).

## Amendment 2026.11 (wave 3) — a continuous switch, far images, the cross-layer path

User decision D2 of 2026-11-15, audits W2B-01/-02, finding 10.28.

1. **Continuous switch.** Decision 3 switched from the exact integral to
   the far form at six segment lengths in one step. The pairs six segments
   apart on a straight wire, and in two-layer soil every segment's own
   Tagg/Sunde image when an image distance equals six segment lengths, sit
   exactly there, so the round-off of the coordinates decided their form:
   the same world at different positions spread by up to 4.4e-7 (a 12 m
   strip) and 2.6e-5 (two layers) in the impedance. Pairs between six and
   7.5 segment lengths now take a C¹ blend of the exact and the far value,
   for direct pairs, every image, the images of the cross-layer correction
   and field points. The spread is at most 1e-13 over local positions and
   the UTM solves lie as close to the local ones as with the point kernel
   (3e-11). Alternatives: a margin on the distance (a new knife-edge at
   another distance, which a chosen $h_1$ hits again) and a far form exact
   to round-off at six segment lengths (about ten more multipole orders);
   the blend costs the integrals of the pairs in the band (+24 % solve time
   in homogeneous soil).
2. **Far form.** The far form of segment pairs carries the radius term
   $-a^2/(2R^3)$ of the reduced kernel $1/\sqrt{R^2 + a^2}$ that every exact
   pair uses; without it the far form was $a^2/(2R^2)$ away from the exact
   pairs it joins, which made straight wires converge non-monotonically at
   the 1e-6 level.
3. **Far Tagg/Sunde images** carry the quadrupole term (the alternative
   above). On every image it cost 2.2 … 3.9 times the solve of 16 house
   grids; the term is therefore tapered off between 60 and 120 segment
   lengths of vertical distance between observer and image (below
   $1/(8\cdot 60^2)$ of the image's point value, the level of the far
   form's remainder at the switch), a property of the pair, and images
   beyond every observer are skipped: 1.2 … 1.5 times, the impedance
   within 1e-6 of the term on every image. Gain: the strip of the
   alternative within 7.7e-7 instead of 5.1e-4, monotone in the segment
   length; its surface potential within 8e-6 instead of 0.94 %; house grids
   3e-6 … 1.8e-4.
4. **The cross-layer path** (ADR-0007) treats the images of its layered
   correction alike (ADR-0007, amendment wave 3).

## Amendment 2026.11 (wave 3, audit W3A) — image pairs, the cost of the far images, a point-form option

The author's decision of 2026-11-15 and audit W3A (stream fw3xl).

1. **Image pairs.** Decision 2 applied the reduced kernel to every pair,
   images included, and point 2 of the wave-3 amendment gave the far form
   of images the radius term. An image line that passes a wire at a
   transverse distance $\rho_\perp \ge a$ acts with its axis value (the
   average of $\ln\lvert x - y\rvert$ over a circle of radius $a$ is
   $\ln\max(\rho, a)$), so the reduced kernel left out $a^2/(2d^2)$ of an
   image $d$ away: a ring 1 cm above a 1000 / 30 Ω·m interface read
   +1.60 % against its exact thin ring (audit W3A-02, finding of w4axi).
   The line potential of an image at a point of the observer now uses
   $\max(\rho_\perp, a)$, which is the reduced kernel for a collinear image
   (a rod against its own image, where it is exact); an image within one
   wire radius of its own segment blends into the reduced kernel of the
   direct pair it nearly coincides with, so that a wire on the interface is
   the same on both sides; the far form of images has no radius term; the
   image corrections are symmetrised (the pointwise rule differs between the
   two orders of a skew pair). The ring is within 0.004 % now, a rod through
   the interface within 1.4e-6 of its exact two-owner value. The soil-surface
   image keeps the reduced kernel (homogeneous results unchanged; with the
   transverse distance a house grid at 0.8 m would move by 4e-7, at 0.1 m by
   1.7e-5 — an open decision).
2. **Cost of the far images.** Every image the longest segment of the world
   reached was evaluated on every pair (audit W3A-01: eight 9.4 m pieces
   among 1920 segments multiplied the far-image work by 5.6), and the image
   density of a high contrast made a 40-house network in 1000 / 30 Ω·m with
   $h_1$ = 1.5 m (82 images within the taper) 4.4 times the solve of the
   base. An image is now evaluated on the pairs whose longer segment reaches
   it (bit-identical), the far-image part of the reaction matrix on its upper
   half (it is symmetric), with one division per evaluation and the taper and
   tolerance tests left out where they hold for a whole block (round-off).
   That network takes 1.8 times the solve before wave 3 with numba (3.6
   times without; 4.2 and 9.7 before the audit), 16 house grids with eight
   long pieces 1.5 times (2.8 before; performance page). Rejected:
   tabulating the far term per image family in the horizontal distance
   (an interpolation error instead of round-off) and multi-threading
   (the base is single-threaded, parameter sweeps run processes in parallel).
3. **Option.** `Engine.image_far_field = "point"` keeps the far images in
   point form: the dense far-field term is left out, the near image pairs
   (within 7.5 segment lengths, a segment's own images included) stay
   Galerkin integrals on the series and the cross-layer path alike, and the
   continuous switch and the transverse distance stay. It is not the result
   before wave 3: within the upper layer up to 2.3e-4 from it (the blend of
   D2). The quadrupole form stays the default. The point form moves
   networks by 3e-6 … 4.3e-4 (16 house grids 0.4 m above a 1000 / 10 Ω·m
   interface), a straight wire near the interface by 4e-4 and the surface
   potential above it by up to 0.8 %, and takes 0.49 of the default's time
   on the 40-house network (0.91 times the solve before wave 3; the network
   solves on the cross-layer path). A common `Engine` field next to
   `image_max_terms` / `image_series_tol` rather than a backend option,
   because `solve_mutual_*` and `FieldResult.potential` read it too; any
   other value raises `ValueError` (audit W3R2A-02).

   Audit W3R2A-01 (round 2): the option first kept, on the cross-layer
   path, the line form of the interface image only, as before wave 3. The
   own surface and transmitted images of a segment below the interface were
   then point sources, so a 75 m strip below a 0.5 m top layer read −1.0e-3
   at a segment length of 2.5 m and −58 % at 37.5 m, and a network with a
   75 m conductor in two pieces +1.1 % against the default. Every image of
   the layered correction now keeps its near pairs, and a segment's own
   images follow the switch of every pair (exact within six segment lengths,
   blended to 7.5, point beyond — the self pair of the series path): the
   strip within 1.1e-5 at every segment length, the network within 4e-6.
   Below thin top layers this costs the near image pairs the option had
   saved: 1.5 … 2.4 times the solve before wave 3 instead of 1.2 (still
   0.51 … 0.94 of the default's time). The auditor's variant (every self
   image at every distance) and the default's tapered diagonal agree with it
   to 2e-6 on networks and cost up to 14 % more.
4. **Wires within a fraction of a radius of the interface** (audit
   W3R2A-03). The image of a wire within $a/20$ of the interface blends into
   the reduced kernel of the direct pair over a tenth of the wire radius
   (before: over one radius, whose blended distance peaked at $a/\sqrt 2$
   and gave a ring 1.75 mm above a 1000 / 30 Ω·m interface +22 %). Z then
   moves through $\pm a$ of the interface with wiggles of at most 0.17 %
   against its direction (strips and rings in 1000 / 30 and 30 / 1000 Ω·m;
   before up to 24 %).

## References

- Review pass 10 (2026-11-07), findings 10.28 and 10.33, decision
  question 4.
- Harrington, R. F. (1968). *Field Computation by Moment Methods*.
  Macmillan, New York.
- Wilton, D. R., et al. (1984). Potential integrals for uniform and linear
  source distributions on polygonal and polyhedral domains. *IEEE
  Transactions on Antennas and Propagation* AP-32(3), 276–281.
- The [kernels page](../explanation/kernels.md) of the documentation.
