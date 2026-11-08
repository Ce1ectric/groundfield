# Soil

Soil models describe the conducting half-space: horizontal layers with
a resistivity $\rho_i$ in Ω·m and a thickness $h_i$ in m, below
non-conducting air. Three models share one interface:
`HomogeneousSoil(rho=...)`, `TwoLayerSoil(rho_1=..., rho_2=..., h_1=...)`
and `MultiLayerSoil(layers=[SoilLayer(rho=..., h=...), ...])` with
`h=None` for the bottom half-space.

## Mathematical and physical context

Below 1 kHz the soil is a quasi-static conductor. The potential obeys

$$
\nabla\cdot\bigl(\rho^{-1}(z)\,\nabla\varphi\bigr) = 0 \quad (z > 0),
\qquad \partial_z\varphi = 0 \ \text{at}\ z = 0,
$$

with $\varphi$ and the normal current density $\rho^{-1}\partial_z\varphi$
continuous at every layer interface and $\varphi \to 0$ at remote earth.
For two layers the image-series coefficient is the reflection
coefficient

$$
K = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}, \qquad |K| < 1,
$$

the source is reflected into a series of images spaced $2h_1$ apart
whose strengths decay like $K^n$ (slowly for high contrast; the AP1
corner soil $\rho_1 = 1000$, $\rho_2 = 30\,\Omega\,\mathrm{m}$ has
$|K| = 0.94$). Three or more
layers are solved by quadrature of the layered Green's function
(`mom_sommerfeld`) or by the experimental `fem` backend; see
[Explanation → Engines](../../explanation/engines/index.md).

**Validity.** Linear, isotropic, horizontally layered soil; the
displacement current is neglected against the conduction current.
Their ratio is $\omega\varepsilon_0\varepsilon_r\rho$:

```python
import math

eps_0 = 8.8541878128e-12                         # F/m
ratio = {rho: 2 * math.pi * 1e3 * eps_0 * 81 * rho for rho in (100.0, 1e3, 1e4)}
assert ratio[100.0] < 1e-3 and ratio[1e3] < 1e-2 and ratio[1e4] < 5e-2
```

At 1 kHz and $\varepsilon_r = 81$ it is $4.5\cdot10^{-4}$ for
100 Ω·m and $4.5\cdot10^{-3}$ for 1000 Ω·m; only very resistive rock
(10 kΩ·m) reaches $4.5\cdot10^{-2}$, a limit of the quasi-static model
itself. No backend reads `relative_permittivity`; a non-default value
warns with `InertConfigurationWarning`.

## Example

```python
import groundfield as gf

homogeneous = gf.HomogeneousSoil(rho=100.0)
two_layer = gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=1.0)  # 1 m topsoil over rock
three_layer = gf.MultiLayerSoil(
    layers=[
        gf.SoilLayer(rho=80.0, h=0.5),
        gf.SoilLayer(rho=300.0, h=2.0),
        gf.SoilLayer(rho=50.0),          # bottom half-space: h=None
    ]
)

K = (two_layer.rho_2 - two_layer.rho_1) / (two_layer.rho_2 + two_layer.rho_1)
assert abs(K - 2 / 3) < 1e-12
assert three_layer.kind == "multi_layer"     # discriminator for JSON round trips
```

Parameters are validated on construction and on assignment: a
resistivity or thickness must be finite and positive. The names
`resistivity` and `thickness` of releases up to 0.15.0 still work with a
`DeprecationWarning` (see [Deprecations](../deprecations.md)).

## API reference

::: groundfield.soil
