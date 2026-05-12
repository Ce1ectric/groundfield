# Concepts

## Position within the software family

```
  groundmeas   ──▶   groundinsight   ◀──   groundfield
  (measurement)      (reduced network            (field model,
                      model)                      PDE reference)
```

`groundfield` is the field-theoretical reference tool. It supplies the
ground truth that `groundinsight` is measured against, and the
analytic baseline against which `groundmeas` data can be evaluated.

## Solver backends

The numerical core is swappable through a backend parameter. Eight
backends share the same data model (`World`, `Electrode`,
`Conductor`, `Source`) and the same Sommerfeld representation of the
layered Green's function — they differ only in **how** the integral
is evaluated:

| Backend          | Suitable for                                          | Method                                                                              |
|------------------|-------------------------------------------------------|-------------------------------------------------------------------------------------|
| `image`          | homogeneous soil                                      | image-charge sum, closed form                                                       |
| `image_2layer`   | 2-layer soil                                          | Tagg/Sunde geometric image-charge series                                            |
| `image_nlayer`   | homogeneous, 2-layer, or multi-layer (dispatcher)     | image-charge dispatcher (delegates to `image` / `image_2layer`; raises for $n\ge3$) |
| `cim`            | any layered                                           | Complex Image Method (closed form via matrix-pencil fit of $\Gamma_1(\lambda)$)     |
| `mom`            | homogeneous or 2-layer                                | Galerkin Method-of-Moments on the closed-form layered kernels                       |
| `mom_sommerfeld` | any layered                                           | Galerkin MoM with direct Sommerfeld quadrature (reference engine, slow)             |
| `bem`            | any layered                                           | Boundary-element collocation with the CIM kernel                                    |
| `fem`            | any layered                                           | Axisymmetric volume PDE with equivalent-hemisphere reduction                        |

For homogeneous cases `image` is the default — evaluation reduces to
a vectorised sum over image sources in NumPy. For 2-layer soils
`image_2layer` provides a closed-form alternative based on the
Tagg/Sunde image series. `Engine.solve` auto-dispatches `"image"` to
`image_2layer` / `image_nlayer` based on the soil model, so notebooks
written for the homogeneous case keep working when the soil is
replaced by a layered one. For $n \ge 3$ layers the closed-form path
runs through `cim`; `mom_sommerfeld` is the absolute reference
engine. The full engine theory is collected in
[Engine theory](engines/index.md); the selection heuristic is
documented in [ADR-0002](adr/0002-engine-family.md).

## Modelling assumptions

- **Quasi-static** in the soil for $f \lesssim 1\,\mathrm{kHz}$. The
  scalar electric potential then satisfies
  $\nabla \cdot (\sigma \nabla \varphi) = 0$.
- **Carson correction** for the earth-return path of overhead and
  buried conductors; no full-wave model. See *Earth-return inductive
  coupling* below.
- **Layered soil** as horizontal half-spaces with piecewise constant
  conductivities.
- **Thin-wire approximation** for electrodes and conductors (Method of
  Moments).

## Earth-return inductive coupling

Distributed conductors carry a per-segment longitudinal-impedance
block whose three pieces correspond to the three physical layers
of the model:

$$
Z_b(\omega) \;=\; \underbrace{R}_{\text{ohmic, ADR-0003}}
   \;+\;\underbrace{j\omega\,L_\text{Neumann}^{\text{(perfect mirror)}}}_{\text{inductive, ADR-0004}}
   \;+\;\underbrace{\Delta Z_\text{Carson}(\omega, \sigma_\text{earth}, h_i, h_j, d_{ij})}_{\text{earth-return, ADR-0005}}.
$$

The first piece is purely resistive — `Conductor.cross_section`
produces $R = \rho_\text{mat} L / A$ per branch, available since
ADR-0003. The second piece is the magnetic-image inductance under
the assumption $\sigma_\text{earth} \to \infty$ (perfect mirror);
the matrix $L_\text{Neumann}$ is built once before the frequency
loop. The third piece is the Carson 1926 finite-conductivity
correction (ADR-0005); it is rebuilt at every frequency because its
kernel depends on $\omega$ through the dimensionless Carson
parameter $a = D\sqrt{\omega\mu_0\sigma_\text{earth}}$, with
$D = 2h_i$ for self and $D = \sqrt{(h_i+h_j)^2 + d_{ij}^2}$ for
mutual.

Two engine-side switches govern this block:

- `Conductor.inductance_model = "neumann"` activates the second
  piece; without it the system is purely real and the historic DC
  fast path is preserved bit-exact.
- `Engine.earth_inductive_model` selects the third-piece model:
    - `"perfect_mirror"` (default, ADR-0004) — no third piece, the
      earth is a perfect magnetic mirror.
    - `"carson_series"` (ADR-0005) — Carson 1926 per-meter formula
      scaled by segment length. Asymptotically correct for long
      parallel wires over homogeneous earth; an approximation for
      short wires or layered soils.
    - `"sommerfeld"` (ADR-0006) — geometric integration of the
      σ-dependent vector-potential Green's function over the
      actual segment-pair geometry, with native support for
      layered earth (Pollaczek/Wait kernel). Rigorous for any
      wire length and orientation; converges to `"carson_series"`
      on the cluster-impedance level for long parallel wires
      over homogeneous earth.

The natural diagnostic for this block is the soil skin depth
$\delta(\omega) = \sqrt{2/(\omega\mu_0\sigma_\text{earth})}
\approx 503\sqrt{\rho_\text{earth}/f}\,\mathrm{m}$, exposed at every
solved frequency through `FieldResult.metadata["penetration_depth"]`.
For AP1 frequencies (≤ 1 kHz) and resistivities (50–5000 Ω·m) the
skin depth ranges from ≈ 350 m to ≈ 35 km — comparable to or larger
than typical TN-Ortsnetz distances, which is exactly why Carson
matters.

## The `rho-f` model

The objective of a field computation in `groundfield` is not only a
numerical result but also a compression of the solution down to a
small set of parametrically readable quantities. For the two-port
case that compression is the `rho-f` model:

$$
Z(\rho, f) = R_{0}(\rho) + X(\rho, f)
$$

where $R_{0}$ is the low-frequency spreading resistance and $X$
collects the frequency-dependent reactive contributions. The
coefficients are fitted against the field solution and end up as
`BusType.impedance_formula` in `groundinsight`.

## Work package 1

`groundfield` is the primary tool for **work package 1**, which
investigates TN distribution networks with substation, house
connections, and cable cabinets in layered soil. The core questions
are:

1. How strongly does a remote current injection influence the
   grounding-measurement result?
2. How important are coupling and return-path effects in the
   low-frequency range?
3. Can robust statements be derived for typical distribution-network
   configurations?

These questions form the first building block of the dissertation;
`groundfield` provides the required numerical basis.
