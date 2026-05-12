# Solver

The solver layer holds the numerical core. Available backends:

| Backend | Soil model | Method | Status |
|---|---|---|---|
| `image` | `HomogeneousSoil` | image-charge sum (point sources + line self-action) | implemented |
| `image_2layer` | `TwoLayerSoil` | Tagg/Sunde image-charge series | implemented |
| `image_nlayer` | `HomogeneousSoil`, `TwoLayerSoil`, `MultiLayerSoil` | image-charge dispatcher (delegates to `image` for `n=1`, to `image_2layer` for `n=2`; raises for `n ≥ 3`) | implemented |
| `cim` | any layered | Complex Image Method (matrix-pencil fit of $\Gamma_1(\lambda)$) | implemented |
| `mom` | `HomogeneousSoil` or `TwoLayerSoil` | Galerkin Method-of-Moments on segment level (independent resolution scheme over the same Green's-function kernels) | implemented |
| `mom_sommerfeld` | any layered | Galerkin MoM with direct Sommerfeld quadrature (reference engine) | implemented |
| `bem` | any layered | Boundary-element collocation with the CIM kernel | implemented |
| `fem` | any layered | Axisymmetric volume PDE with equivalent-hemisphere reduction | implemented |

## Mathematical / physical model

For a point current source $I$ at $z_s$ in the upper layer of a
horizontally stratified half-space (insulating soil surface at
$z = 0$), every backend evaluates the same quasi-static Sommerfeld
representation of the potential:

$$
\varphi(s, z) \;=\; \frac{\rho_1\, I}{4\pi}
\int_0^{\infty} \bigl[
  e^{-\lambda |z - z_s|}
+ \Gamma_1(\lambda)\, e^{-\lambda (z + z_s)}
\bigr]\, J_0(\lambda s)\, d\lambda,
$$

with the upward-looking reflection $\Gamma_1(\lambda)$ built
recursively from the bottom up
(`groundfield.solver._layered.reflection_gamma`). The engines differ
only in **how** they evaluate this integral:

- closed-form real images (`image`, `image_2layer`, `image_nlayer`);
- closed-form complex images (`cim`);
- direct numerical quadrature (`mom_sommerfeld`);
- volume PDE (`fem`).

ADR-0002 (`docs/adr/0002-engine-family.md`) records the selection
heuristic between the engines.

## Auto-dispatch

`Engine.solve` automatically forwards `backend="image"`:

- to `image_2layer` if `world.soil` is a `TwoLayerSoil`;
- to `image_nlayer` if `world.soil` is a `MultiLayerSoil`.

Notebooks therefore keep working unchanged when the soil model is
swapped.

## Example

```python
import groundfield as gf

# Build a small world (single ring electrode in two-layer soil).
soil  = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=2.0)
world = gf.create_world(soil=soil)
gf.create_electrode(
    world, "ring", name="g1",
    center=(0.0, 0.0, 0.8), radius=5.0, wire_radius=0.005,
)
gf.create_source(world, attached_to="g1", magnitude=1.0)

# Create an engine and solve. Auto-dispatch hands `image` over to
# `image_2layer` because the soil is two-layer.
engine = gf.create_engine(
    backend="image",
    frequencies=[50.0, 150.0, 250.0],
    segment_length=0.05,
)
result = world.solve(engine)

# Cluster impedance of the ring electrode.
print(result.cluster_impedance("g1"))
```

The same `World` can be solved with any of the eight backends —
`compare_engines(world, engines={"image": ..., "mom": ...})` reports
their cluster-impedance agreement (cross-validation rules below).

## Cross-engine validation

`groundfield.compare_engines(world, engines={...})` runs the same
world through several engines and reports the cluster-impedance
agreement. The same rule is enforced by
`tests/test_cross_engines_extended.py`:

- For homogeneous worlds every engine must agree to within 5 %
  (10 % for `fem`).
- For 2-layer worlds the closed-form / image / MoM engines must
  agree with each other to within 5 %.
- A monotone $\rho_2$ sweep at fixed $\rho_1$ must produce a
  monotonically increasing cluster impedance for every engine.

::: groundfield.solver
