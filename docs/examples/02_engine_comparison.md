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

Each backend declares the soil models it accepts, and
`Engine.solve` enforces that contract with a `TypeError` rather
than silently reinterpreting the soil. `image_2layer` in
particular requires a `TwoLayerSoil` instance — it will *not*
take a `HomogeneousSoil`. The physically homogeneous limit is
reached from the two-layer side by setting
$\rho_2 = \rho_1$ and any $h_1$: the reflection factor
$K = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)$ vanishes, every term of
the Tagg/Sunde image series except the direct one drops out, and
the result must reproduce `image` to machine precision. The
comparison loop below therefore pairs each backend with a soil
object it accepts.

## Code

```python
import groundfield as gf

def build_world(soil) -> "gf.World":
    """Two bonded rods on the given soil model."""
    world = gf.create_world(soil=soil)
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

homogeneous = gf.HomogeneousSoil(resistivity=100.0)
# Degenerate two-layer stack: rho_2 = rho_1 => K = 0, the image
# series collapses to the homogeneous case.
degenerate_2layer = gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=2.0)

cases = [
    ("image", homogeneous),
    ("image_2layer", degenerate_2layer),
    ("mom", homogeneous),
    ("bem", homogeneous),
    ("fem", homogeneous),
]
for backend, soil in cases:
    world = build_world(soil)
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

## What you should see

`image` and `image_2layer` agree bit-for-bit — that is the
$K = 0$ collapse of the Tagg/Sunde series, and it is the cheapest
available self-check on the two-layer kernel. The
integral-equation engines (`mom`, `bem`) land within a few tenths
of a percent of the closed-form value; the residual is the
difference between the closed-form average-potential self-term and
the Galerkin / collocation quadrature at this `segment_length`.
`fem` is the outlier by a few percent, and legitimately so: it
reduces the geometry to an axisymmetric equivalent hemisphere, so
a two-rod cluster with a bond conductor is outside the geometry
class it represents exactly. Halving `segment_length` shrinks the
`mom` / `bem` residual but not the `fem` one, which is a modelling
rather than a discretisation error.

`mom_sommerfeld` is the absolute multi-layer reference; the
closed-form `image` family is faster for homogeneous / 2-layer
problems and should be preferred in parameter sweeps that don't
need the full Sommerfeld accuracy.

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
