# ADR-0001: Numerical method for 2-layer soil

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-04-30 |
| **Deciders** | Project maintainers |
| **Scope** | `groundfield` |

## Context

The homogeneous image-charge backend (`backend="image"`) is in place
and validated against Dwight 1936 (< 10 % deviation across rod, rod
pair, ring electrode; a systematic 4–5 % from the midpoint
point-source approximation, decreasing with finer `segment_length`).

The next step is the **2-layer model**: a variable upper layer
($\rho_1$, $h_1$) over a semi-infinite lower layer ($\rho_2$).

Requirements:

- **Frequency range** $f < 1\,\text{kHz}$ → quasi-static; a real
  Green's function suffices.
- **Geometry coverage**: rod, ring, mesh / foundation
  electrode, horizontal connection conductor, auxiliary electrodes —
  i.e. everything the `image` backend already discretises.
- **Self-validation must be possible**: engine A vs. engine B on the
  same set-up. In the limit $\rho_1 = \rho_2$ every 2-layer engine
  must reproduce the Dwight values.
- **Notebook-friendly**: `eng.solve(world)` should return in seconds
  to a few minutes; otherwise parameter studies become impractical.
- **Open source**, **Python 3.12**, no commercial software.

## Decision

We build **two independent engines** for 2-layer soil in parallel:

1. **`backend="image_2layer"`** — image-charge series after
   **Tagg / Sunde**. The fast default path; closed form.
2. **`backend="mom_sommerfeld"`** — Method of Moments with
   **numerical Sommerfeld quadrature** for the Green's function.
   Heavier, but methodologically independent.

`gf.compare_engines(world, [...])` cross-validates them. In the
homogeneous limit (e.g. $\rho_2 = \rho_1$) both must reproduce the
existing `image` backend and therefore Dwight 1936.

## Options considered

### Option A — Image-charge series (Tagg / Sunde)

A point current source at depth $z_s$ in the upper layer produces a
potential field that can be expressed as an **infinite series** of
real and mirrored point sources:

$$
\varphi(\mathbf{r}) = \frac{\rho_1\,I}{4\pi}\sum_{n=0}^{\infty} K^n\,
\Bigl(\frac{1}{r_n^{(1)}} + \frac{1}{r_n^{(2)}}
+ \frac{1}{r_n^{(3)}} + \frac{1}{r_n^{(4)}}\Bigr),
$$

with the reflection coefficient $K = (\rho_2 - \rho_1)/(\rho_2 +
\rho_1)$ and the four image distances $r_n^{(k)}$ that follow from
recursive mirroring at the soil surface and the layer interface.

| Dimension | Assessment |
|---|---|
| Accuracy | Closed form, exponential convergence in $\|K\|$. For $\|K\| \le 0.99$ ~20-50 terms reach $10^{-4}$. |
| Complexity | Very low — same code path as the `image` backend, only the kernel changes. |
| Cost | Few hours of implementation + tests. |
| Scaling | Trivial: $O(N \cdot N_\text{img})$ per field point — like `image`, plus a constant factor. |
| External deps | NumPy only (already required). |
| Validation path | Limit $\rho_1 = \rho_2 \Rightarrow K = 0$ leaves only the $n=0$ term ⇒ exact reduction to the homogeneous backend, automatic Dwight comparison. |
| Sources | Tagg 1964 *Earth Resistances*; Sunde 1968 *Earth Conduction Effects*; IEEE Std 80-2013. |

**Pros:**

- Closed form, **no** numerical quadrature error.
- Implementation is a small extension of the existing kernel.
- Practically free to evaluate during typical parameter sweeps.

**Cons:**

- Convergence issues for $|K| \to 1$ (extreme contrast, e.g. wet clay
  on rock). A series-truncation strategy is needed.
- Cannot directly extend to $n > 2$ layers — for general layered
  models Sommerfeld is the right approach.

### Option B — MoM with Sommerfeld quadrature

The Green's function for a layered half-space is a Hankel transform:

$$
G(\rho, z; z_s) = \frac{1}{4\pi}\int_0^\infty
\Bigl(e^{-|z-z_s|\lambda} + R(\lambda)\,e^{-(z+z_s)\lambda}\Bigr)
J_0(\lambda \rho)\,d\lambda,
$$

with $R(\lambda)$ derived from the layer reflection ratios. Evaluation
through numerical quadrature (e.g. Gauss-Bessel, or Anderson 1979 /
Talman). Around it: a Galerkin MoM on the discretised electrode
surfaces.

| Dimension | Assessment |
|---|---|
| Accuracy | Very high (controllable through quadrature tolerance). |
| Complexity | High: oscillating Bessel kernels, semi-infinite integration, singularity handling. |
| Cost | 1–2 weeks of careful implementation + convergence studies. |
| Scaling | $O(N^2)$ MoM matrix; per-entry cost of $G$ higher than 1/r ⇒ larger constant factor. Caching $G$ on a tabulated grid keeps it tractable. |
| External deps | NumPy + SciPy (already required). |
| Validation path | Limit $\rho_1 = \rho_2 \Rightarrow R(\lambda) = 0$ ⇒ exact reduction to the 1/r kernel ⇒ Dwight. |
| Sources | Sommerfeld 1909 / 1949; Visacro 2007; Chow & Salama 1989. |

**Pros:**

- Generic approach, **directly extendable** to $n$ layers and later
  to frequency-dependent soil (Visacro / Alipio).
- Provides correct **self-action** through line integrals instead of
  the midpoint approximation. This removes the residual ~5 % bias
  against Dwight that the current `image` backend carries.
- Methodologically independent from option A → ideal as a
  cross-validation engine.

**Cons:**

- Implementation effort is high; convergence of slowly oscillating
  integrands is fragile.
- Without caching / interpolation the runtime is significantly
  larger.
- Singularity handling at $\lambda \to 0$ and $\lambda \to \infty$
  must be done carefully.

### Option C — Complex-image method (Chow / Salama)

The exact Green's function is approximated by a **finite sum of
complex point sources**:

$$
G(\mathbf{r}) \approx \frac{\rho_1}{4\pi}\sum_{k=1}^{M}
\frac{a_k}{|\mathbf{r} - \mathbf{r}_k^{(\mathrm{complex})}|}.
$$

The $a_k, \mathbf{r}_k$ are obtained from a Prony / matrix-pencil
fit to exact Sommerfeld values.

| Dimension | Assessment |
|---|---|
| Accuracy | Medium to high, depending on $M$. Typically $M = 5\dots 15$. |
| Complexity | Medium: a robust Prony fit is non-trivial. |
| Cost | 1–2 weeks. |
| Scaling | Very fast: after the fit the evaluation reduces to a point-source sum. |
| Validation path | Indirect; needs cross-check against Sommerfeld. |

**Pros:**

- Fast as the image-charge series, but supports $n$ layers.

**Cons:**

- Approximation character; fit quality varies with geometry.
- Adds non-trivial complexity (Prony) without a clear win for typical cases —
  Tagg / Sunde is enough for two layers, and beyond two layers
  Sommerfeld is the more robust choice anyway.

### Option D — FEM (3-D volume)

Half-space box meshed with `scikit-fem`, electrodes as embedded wire
networks.

| Dimension | Assessment |
|---|---|
| Accuracy | Very high with fine meshes; limited by discretisation error. |
| Complexity | High: 3-D mesh, truncation boundary, mesh adaptation. |
| Cost | 3–4 weeks and up. |
| Scaling | Poor: a 30 × 30 × 30 m box at 0.1 m resolution gives ~$10^7$ tetrahedra. |
| Validation path | Direct against all other engines; primarily useful as a "heaviest hammer". |

**Pros:**

- Maximum generality (arbitrary inhomogeneities, voids, rock blocks).

**Cons:**

- Overkill for typical cases; only justified once heterogeneity varies in space
  in a way that layered models cannot capture.
- The artificial truncation boundary introduces its own errors that
  are hard to separate from the actual model.

### Option E — IEEE Std 80 approximations

Integral / mean-potential formulas for mesh and mesh-with-rod
electrodes ($R = R_g + R_t + R_m$). Only works for the **standard
geometries** covered by IEEE Std 80 (rectangular meshes with uniform
spacing).

**Pros:** quick, widely used in industry.
**Cons:** no generality, no value-add for the research work.

## Trade-off analysis

The question "one engine or two?" is decided by the research goal.
With only one backend in `groundfield`, every systematic deviation
(model error, sign error in the reflection coefficient, quadrature
error) **cannot be detected internally** — only through comparison
against literature reference cases and ultimately against
`groundinsight` as the downstream consumer. The modelling literature
(Visacro 2017; Alipio 2014) shows that such errors are common in
solver implementations and only surface through cross-method
comparisons.

Therefore: **two engines, distinct method families, both validated
against Dwight in the homogeneous limit.** Tagg / Sunde is the
natural fast engine; Sommerfeld is the natural second one. The
combination is also the established practice in the modelling
literature (Visacro, Alipio, Cooray).

C and D would be additional engines that we may add later if needed
— recorded as roadmap items but not on the current plan.

## Consequences

### What becomes easier

- **Self-validation**: any world can be solved with both engines and
  compared automatically.
- **typical parameter sweeps** run on the fast engine A; spot checks run
  on engine B as a safety verification.
- **Frequency-dependent soil models** (Visacro, Alipio) can later be
  bolted onto the Sommerfeld engine directly.

### What becomes harder

- Two implementations to keep in sync. The data model
  (`World`, `FieldResult`) stays identical though — only the backend
  swaps.
- More tests, more CI time. With ~40 tests today this is bearable.
- Cross-engine comparison needs clearly defined per-geometry
  tolerances.

### What we may revisit later

- **Convergence limit of Tagg / Sunde** at $|K| \to 1$: should typical
  ever hit such soil contrasts and the series become unstable, we
  switch to Sommerfeld. The engine logic does this automatically
  (series-truncation control with fallback warning).
- **Complex-image method** as a third engine, if we need more than
  two layers and Sommerfeld is too slow.

## Action items

1. [x] **Engine A** (`backend="image_2layer"`) implemented as an
   extension of `solver/image.py` — same discretisation, Tagg / Sunde
   image-charge series. **Truncation:** $|K|^n < 10^{-6}$ or
   $n_\text{max} = 100$. Diagnostics in
   `FieldResult.metadata`. Auto-dispatch in `Engine.solve`:
   `TwoLayerSoil` switches transparently.
   *(2026-04-30)*

2. [x] **Engine B** (`backend="mom"`) implemented in
   `solver/mom.py`. Picked the lighter pragmatic route over the
   originally planned Sommerfeld-quadrature variant:

   - **Done now**: Galerkin MoM on segment level using the *same*
     Green's-function kernel as the matching image backend
     (`HomogeneousSoil` shares the kernel with `image`,
     `TwoLayerSoil` shares the Tagg/Sunde series with
     `image_2layer`). Builds the full N×N reaction matrix and
     solves an (N+K)×(N+K) linear system for the per-segment
     currents and the cluster potentials, instead of assuming
     uniform per-unit-length currents. This already removes the
     uniform-current bias of the image backends — the resolution
     scheme is methodologically different.
   - **Not done**: independent Sommerfeld quadrature for the
     Green's function. The current `mom` backend therefore shares
     the *physics* with the image backends; the cross-validation
     it provides is on the resolution scheme, not on the
     Green's-function evaluation. A future Sommerfeld-quadrature
     variant remains on the roadmap when we either need more than
     two layers or want a fully independent layered Green's
     function.

   Cross-engine tests pass within ~1 % on cluster impedances and
   surface potentials for both homogeneous and 2-layer worlds.
   *(2026-04-30)*

3. [x] **`gf.compare_engines(world, [eng1, eng2], rel_tol=...)`**
   implemented as a top-level helper. Returns an `EngineComparison`
   report with cluster impedances per engine, max deviation, stub
   detection, and an optional point-sample.
   *(2026-04-30)*

4. [x] **pytest suite extended** with `tests/test_two_layer.py` —
   limit $\rho_1 = \rho_2$ reproduces the `image` backend exactly
   (difference = 0); sign behaviour of K, auto-dispatch, series
   convergence and cross-engine sanity are all covered.
   *(2026-04-30)*

5. [x] **Sanity tests at small $|K|$** ($\rho_1 = 100$,
   $\rho_2 = 110$ ⇒ $|K| \approx 0.048$): `image_2layer` deviates by
   < 5 % from the homogeneous result — matches expectations.
   *(2026-04-30)*

6. [x] **Notebook `02_two_layer.ipynb`** with a parameter sweep over
   $K \in [-0.82, +0.82]$ (rho_2 from 10 to 1000) and over the
   layer thickness $h_1 \in \{1.55, 2, 3, 5, 10, 20\}$ m. Plots of
   the cluster impedance, trumpet comparison, and the K=0 sanity
   check.
   *(2026-04-30)*

Open: full Sommerfeld-quadrature variant of engine B — see action
item 2 (current `mom` backend covers the resolution-scheme
cross-check; the Green's-function cross-check is deferred).

## References

- Dwight, H. B. (1936). *Calculation of Resistances to Ground*. AIEE
  Transactions, **55**, 1319–1328. — Primary source for the
  homogeneous limit, available in the repo under
  `groundfield.references.dwight1936`.
- Tagg, G. F. (1964). *Earth Resistances*. Pitman.
- Sunde, E. D. (1968). *Earth Conduction Effects in Transmission
  Systems*. Dover.
- Sommerfeld, A. (1909). *Über die Ausbreitung der Wellen in der
  drahtlosen Telegraphie*. Annalen der Physik, **28**, 665–736.
- Chow, Y. L., Salama, M. M. A. (1989). *A simplified method for
  calculating the substation grounding grid resistance*. IEEE Trans.
  PAS, **104**(2), 379–386.
- Visacro, S. (2007). *A comprehensive approach to the grounding
  response to lightning currents*. IEEE Trans. PD, **22**(1),
  381–386.
- Alipio, R., Visacro, S. (2014). *Frequency dependence of soil
  parameters: effect on the lightning response of grounding
  electrodes*. IEEE Trans. EMC, **56**(1), 132–139.
- IEEE Std 80-2013. *Guide for Safety in AC Substation Grounding*.
