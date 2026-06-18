# 02 — Comparison of the solver engines

`groundfield` ships several solver backends with different
trade-offs between speed, soil-model coverage, and modelling
fidelity. The fastest way to build trust in any one of them is to
run the same physical problem through several engines and inspect
the agreement.

## Engines at a glance

| Backend             | Soil model                | Approach                                    |
|---------------------|---------------------------|---------------------------------------------|
| `image`             | homogeneous               | Image-charge sum, closed-form                |
| `image_2layer`      | two-layer                 | Tagg/Sunde image-charge series              |
| `image_nlayer`      | dispatcher                | Auto-routes by soil layer count             |
| `cim`               | multi-layer               | Complex Image Method (matrix pencil)        |
| `mom`               | homogeneous + 2-layer     | Galerkin Method of Moments                  |
| `mom_sommerfeld`    | multi-layer               | Direct Sommerfeld quadrature (reference)    |
| `bem`               | homogeneous               | Boundary-Element / collocation              |
| `fem`               | axisymmetric              | Finite-element (axisymmetric mesh)          |

The engine theory pages (`docs/engines/...`) document each kernel
in depth. Here we focus on the *user-side* comparison loop.

## Test geometry

A single bonded-rod cluster — two 2 m driven rods 4 m apart,
joined by a bare-copper bond conductor — sitting on a homogeneous
soil with $\rho = 100\,\Omega\,\mathrm{m}$. The cluster has a
single current source at 1 A, returning through remote earth. We
evaluate `cluster_impedance` at 50 Hz on every engine.

## Code

```python
import groundfield as gf

def build_world() -> "gf.World":
    """Two bonded rods on homogeneous soil."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="rod_a",
                        position=(0.0, 0.0, 0.5), length=2.0,
                        wire_radius=0.01)
    gf.create_electrode(world, "rod", name="rod_b",
                        position=(4.0, 0.0, 0.5), length=2.0,
                        wire_radius=0.01)
    gf.create_conductor(world, name="bond",
                        start="rod_a", end="rod_b",
                        conductor_type="bare_copper")
    gf.create_source(world, attached_to="rod_a", magnitude=1.0)
    return world

backends = ["image", "image_2layer", "mom", "bem", "fem"]
for backend in backends:
    world = build_world()
    engine = gf.create_engine(
        backend=backend,
        segment_length=0.5,
        frequencies=[50.0],
    )
    result = engine.solve(world)
    Z = result.cluster_impedance("rod_a")[0]
    print(f"{backend:<16s} |Z| = {abs(Z):7.3f} Ohm "
          f"(R = {Z.real:6.3f}, X = {Z.imag:+6.3f})")
```

`image_2layer` accepts a homogeneous soil too — the upper layer
resistivity equals the lower one and the Tagg/Sunde series collapses
to the homogeneous case, so the result must match `image`.

## What you should see

Within their validity envelopes the engines agree to better than
1 % on this geometry. Larger residuals point at a discretisation
that is too coarse for the highest-resolution engine; halve
`segment_length` and re-run. `mom_sommerfeld` is the absolute
multi-layer reference; the closed-form `image` family is faster
for homogeneous / 2-layer problems and should be preferred in
parameter sweeps that don't need the full Sommerfeld accuracy.

## Picking a backend

- Quick exploration, homogeneous soil → `image`.
- Typical 2-layer studies → `image_2layer` (default for any
  `TwoLayerSoil` world).
- More than two layers → `image_nlayer` dispatcher, or `cim` for
  many frequencies.
- Reference values for paper-grade validation → `mom_sommerfeld`.
- Vertical cross-sections at non-trivial mesh refinement → `bem`
  or `fem`.

## Where to go next

- Add a real two-layer soil and re-run the loop — see
  [03 — Multi-layer soil models](03_multilayer_soil.md).
- Visualise the resulting potential field via every backend — see
  [09 — All plots in action](09_plot_gallery.md).
