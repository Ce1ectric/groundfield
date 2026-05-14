# `mom` — Galerkin Method-of-Moments

## Physical context

The image-charge backends (`image`, `image_2layer`, `image_nlayer`,
`cim`) all share an additional simplifying assumption: the current
distribution along an electrode is **uniform per unit length**.
This assumption is convenient — it lets the cluster constraints be
written directly on the per-electrode total current — but it is a
genuine approximation. The true current distribution along a thin
wire embedded in soil is non-uniform: it is concentrated at the
electrode ends and at junctions, with a residual that scales as
$\sim 5\,\%$ of the integrated input impedance for the canonical
Sunde rod.

The Galerkin Method-of-Moments (MoM) drops this assumption. It
solves a full segment-level linear system for the actual
per-segment currents, and only enforces the cluster
equipotentiality and sum-of-currents constraints as augmenting
rows. The kernel is the **same** layered Green's function as the
matching image backend (homogeneous for $n = 1$, Tagg / Sunde for
$n = 2$); only the resolution scheme differs.

`mom` was introduced as the second engine in ADR-0001 to provide
an *independent resolution scheme* against which the image
backends could be cross-validated. The agreement was deliberately
designed to be tight (within 2 %) precisely because the kernel is
identical — any larger discrepancy would point to a bug in either
the cluster constraint logic or the kernel evaluation.

## Governing equation: boundary integral

A wire-segment system in soil satisfies the boundary integral
equation

$$
\varphi(\mathbf{r}_i) \;=\; \frac{1}{4\pi}
\sum_{j=1}^{N} I_j \int_{\Sigma_j} G(\mathbf{r}_i, \mathbf{r}'_j)
\,dS_j,
$$

with $G$ the layered Green's function (the same one the image
backends use), $\Sigma_j$ the surface of segment $j$, and
$\varphi(\mathbf{r}_i)$ the (unknown) cluster potential of the
cluster containing segment $i$. In matrix form, $Z\,I = \varphi$
with the **reaction matrix**

$$
Z_{ij} \;=\; \frac{\rho_1}{4\pi} \cdot
\frac{1}{L_i} \int_{\text{seg}_i} \! \int_{\Sigma_j}
G(\mathbf{r}, \mathbf{r}')\, dS_j\, d\ell.
$$

The Galerkin formulation arises by averaging $\varphi$ over segment
$i$ — i.e. taking the same basis function as the test function.
This is one of two natural choices; the alternative (point
matching at the segment midpoint) is implemented in
[`bem`](bem.md).

## Numerical strategy

### Reaction matrix assembly

The reaction matrix is built directly from the same self-kernel
factory that the image backend uses:

- For `HomogeneousSoil` it calls
  `_self_corrected_kernel(seg_points, seg_lengths, wire_radii,
  identity, rho)` — feeding the identity matrix as the "currents"
  argument extracts the full $N \times N$ kernel matrix in one
  vectorised pass.
- For `TwoLayerSoil` it calls
  `_two_layer_self_kernel_factory(soil, max_terms, tol)` — same
  trick, but with the Tagg / Sunde geometric series.

The diagonal of $Z$ carries the analytical line self-potential
$2 \ln(L_i / a_i) / L_i \cdot \rho_1 / (4\pi)$ plus the layered
image contributions. The off-diagonals use the point-source
approximation (segment midpoints) for the $1/r + 1/r_{\text{air}}$
direct + air-mirror part and for every $K_1^n$ image term.

### Galerkin solve

The cluster constraints are enforced through an augmented system

$$
\begin{bmatrix} Z & -C \\ C^{\top} & 0 \end{bmatrix}
\begin{bmatrix} I_{\text{seg}} \\ \varphi_c \end{bmatrix}
= \begin{bmatrix} 0 \\ I_{c,\text{in}} \end{bmatrix},
$$

with $C$ the segment-to-cluster membership matrix
($C_{ic} = 1$ if segment $i$ belongs to cluster $c$, else 0), and
$I_{c,\text{in}}$ the input current of cluster $c$. The first $N$
rows enforce $\varphi_i = \varphi_c \;\forall i \in c$ on the
reaction-matrix level; the next $K$ rows enforce $\sum_{i \in c} I_i = I_{c,\text{in}}$ for every active cluster.

The system is **symmetric** (since $Z$ is symmetric and $C$ enters
once with each sign); `numpy.linalg.solve` handles it directly.

Inactive segments (those whose cluster has zero input current) are
excluded from the unknowns to keep the matrix small.

### Real / imaginary split

Because $Z$ is real and the only complex contribution to the
right-hand side is the source phase, the system is solved
separately for the real and imaginary part of every cluster
current. This avoids complex arithmetic in the LU solve.

## Validity envelope

| Property | Range / value |
|---|---|
| Soil model | `HomogeneousSoil` or `TwoLayerSoil` |
| Frequency | quasi-static, $f < 1\,\text{kHz}$ |
| Electrode placement | every segment in the upper layer (for `TwoLayerSoil`) |
| Wire radius / segment ratio | thin-wire, $a \ll L_i$ |
| Mesh size $N$ | $\le 1000$ at acceptable runtime |

For $n \ge 3$ soils, `mom` does not currently apply — the
multilayer kernel is delegated to `cim`, `mom_sommerfeld`, or
`bem`. ADR-0002 keeps the option of extending `mom` with the CIM
kernel as a future action item, but the existing
`mom_sommerfeld` engine already covers that role.

## Convergence and cost

- **Resolution accuracy.** The Galerkin scheme typically gets
  within 1 % of the Dwight reference at the canonical 1.5 m / 5 mm
  rod, eliminating the 4–5 % uniform-current bias of the image
  backend. The improvement carries over to layered soils: `mom` is
  consistently the closest match to the literature reference values
  among the segment-discretised engines.
- **Computational cost.** $O(N^2)$ matrix build, $O((N + K)^3)$ LU
  solve. For typical cases with $N \le 1000$ the runtime is dominated by the
  matrix build (a few hundred milliseconds in the 2-layer case
  with $\sim 200$ image terms).
- **Reduction.** At $K_1 = 0$ the engine collapses bit-exactly to
  the homogeneous Galerkin solution.

## Cross-validation notes

| Counterpart | Expected agreement | What is checked |
|---|---|---|
| `image` (homogeneous) | $\le 2\,\%$ | uniform-current vs. Galerkin |
| `image_2layer` ($n = 2$) | $\le 2\,\%$ | same kernel, different test function |
| `mom_sommerfeld` ($n = 2$) | $\le 5\,\%$ | quadrature reproduces the same physics |
| `bem` (any $n$) | $\le 5\,\%$ | Galerkin vs. collocation on the same kernel |
| Sunde / Dwight closed forms | $\le 5\,\%$ | tighter bound than the image backend |

`mom` is the **independent resolution-scheme reference** in the
cross-validation matrix. When `image` and `mom` agree on a given
problem, the kernel is implemented correctly; when they disagree,
the disagreement is reproducible and points to a uniform-current
bias issue.

## References

- **Harrington, R. F.** (1968). *Field Computation by Moment
  Methods*, Macmillan. The MoM textbook.
- **Meliopoulos, A. P. S., Xia, F., Joy, E. B., Cokkinides, G. J.**
  (1993). An advanced computer model for grounding system analysis.
  *IEEE PWRD* 8(1). MoM for grounding systems with cluster
  constraints.
- **Sunde, E. D.** (1968). *Earth Conduction Effects in Transmission
  Systems*, Dover, sect. 3.5. The line self-potential
  ($2 \ln(L/a) / L$) used on the diagonal.
- **Dawalibi, F. P. & Barbeito, N.** (1991). Measurements and
  computations of the performance of grounding systems buried in
  multilayer soils. *IEEE PWRD* 6(4). Reference values for
  cross-validation.

## Related material

- API reference: `groundfield.solver.mom`.
- ADR-0001 — original methodology decision.
- Notebook `03_cross_engine.ipynb` — image vs. mom side-by-side on
  homogeneous and 2-layer worlds.
