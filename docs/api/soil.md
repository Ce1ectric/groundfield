# Soil

Soil models supply the propagation medium for the field
computation. Every backend in :mod:`groundfield.solver` consumes a
:class:`SoilModel` instance — currently :class:`HomogeneousSoil`,
:class:`TwoLayerSoil` and :class:`MultiLayerSoil`.

## Physical background

Below 1 kHz the soil is a quasi-static conductor: displacement
currents are negligible (the dielectric relaxation time
$\tau = \varepsilon / \sigma$ of moist soil is on the order of
100 ns) and Laplace's equation
$-\nabla \cdot (\sigma \nabla \varphi) = 0$ governs the potential
field. Each soil class therefore reduces to a stack of horizontal
layers with constant resistivity $\rho_i$ and layer thickness
$h_i$. The half-space at infinity acts as a Dirichlet boundary
("remote earth") and the air–soil interface as a Neumann
boundary ($\partial_z\varphi = 0$ at $z = 0$).

For the AP1 dissertation case the two-layer model is the primary
workhorse: a top layer of resistivity $\rho_1$ and finite thickness
$h_1$ over a half-space of resistivity $\rho_2$. The homogeneous
case is recovered for $\rho_1 = \rho_2$; the multilayer case
extends the same matching scheme to arbitrarily many layers (used
by :mod:`groundfield.solver.image_nlayer` and the layered
Sommerfeld solvers).

## Example

```python
import groundfield as gf

# Homogeneous half-space — fastest, used as a sanity baseline.
homo = gf.HomogeneousSoil(resistivity=100.0)

# Two-layer soil — primary AP1 case (e.g. 1 m topsoil over rock).
two_layer = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=1.0)

# Multi-layer soil — generic n-layer stack.
multi = gf.MultiLayerSoil(layers=[
    gf.SoilLayer(rho=80.0, h=0.5),
    gf.SoilLayer(rho=300.0, h=2.0),
    gf.SoilLayer(rho=50.0),  # bottom half-space (no thickness)
])

world = gf.create_world(soil=two_layer)
```

The discriminator field ``kind`` allows soil instances to be
serialised to JSON and reconstructed without prior knowledge of
the concrete subtype.

## API reference

::: groundfield.soil
