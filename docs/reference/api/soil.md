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

## Effective resistivity

A `groundinsight` bus carries one resistivity, its
`specific_earth_resistance`. The AP1 catalogue of real LV networks
describes a two-layer soil for that purpose by its effective
resistivity, a weighted geometric mean of the two layers,

$$
\rho_\text{eff} = \rho_1^{\,1-w}\,\rho_2^{\,w}
\quad\Longleftrightarrow\quad
\ln\rho_\text{eff} = (1 - w)\ln\rho_1 + w\ln\rho_2,
\qquad
w = w_0\left(\frac{h_1}{h_\text{ref}}\right)^{-\beta},
$$

with $h_\text{ref} = 10$ m and the AP1 calibration $w_0 = 0.13$,
$\beta = 0.8$ (the AP2 hand-over carries group-specific values; pass
them as `w0=` and `beta=`). The weight of the lower layer falls with the
thickness of the upper layer: $w$ = 0.47, 0.23, 0.13 and 0.054 for
$h_1$ = 2, 5, 10 and 30 m. AP2 gives a bus $\rho_\text{eff}$, so the
`rho` of a `BusType` fitted in layered soil is $\rho_\text{eff}$ of the
soil it was fitted in ([IO and groundinsight](io.md)).

`gf.effective_resistivity(soil)` (or with the keywords `rho_1`, `rho_2`,
`h_1`)

- returns $\rho$ itself for homogeneous soil and for a stack of equal
  layers, without round-off (the plain formula gives
  200.00000000000003 Ω·m for $\rho_1 = \rho_2 = 200\ \Omega\,\mathrm{m}$,
  $h_1 = 10$ m);
- reduces a `MultiLayerSoil` to its canonical stack (adjacent layers of
  equal resistivity merged, as every layered backend does) and raises
  `ValueError` for three or more distinct layers;
- limits $w$ to $[0, 1]$, so $\rho_\text{eff}$ lies between $\rho_1$ and
  $\rho_2$. Without the limit $w > 1$ for
  $h_1 < h_\text{ref}\,w_0^{1/\beta} = 0.78$ m and $\rho_\text{eff}$
  leaves the interval (17.9 Ω·m for $\rho_1 = 100$,
  $\rho_2 = 30\ \Omega\,\mathrm{m}$, $h_1 = 0.5$ m); with the limit such a
  thin upper layer gives $\rho_2$;
- is bit-identical, inside the calibration range, to
  `r1 ** (1 - w) * r2 ** w` of the AP1 / AP2 scripts.

```python
soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=10.0)
rho_eff = gf.effective_resistivity(soil)               # w = w0 = 0.13 at h_1 = 10 m
assert abs(rho_eff - 1000.0**0.87 * 30.0**0.13) < 1e-9 and abs(rho_eff - 633.907) < 1e-3

assert gf.effective_resistivity(gf.HomogeneousSoil(rho=200.0)) == 200.0
assert gf.effective_resistivity(rho_1=200.0, rho_2=200.0, h_1=10.0) == 200.0
assert gf.effective_resistivity(rho_1=100.0, rho_2=30.0, h_1=0.5) == 30.0  # w limited to 1
assert gf.effective_resistivity(soil, w0=0.0) == 1000.0  # w0 = 0: the upper layer

stack = gf.MultiLayerSoil(    # two-layer after merging: rho_1 = 100, h_1 = 2 + 3 m
    layers=[gf.SoilLayer(rho=100.0, h=2.0), gf.SoilLayer(rho=100.0, h=3.0), gf.SoilLayer(rho=30.0)]
)
assert gf.effective_resistivity(stack) == gf.effective_resistivity(rho_1=100.0, rho_2=30.0, h_1=5.0)

try:
    gf.effective_resistivity(three_layer)    # 80 / 300 / 50 Ohm m from the example above
except ValueError as error:
    print(error)    # effective_resistivity: the soil has 3 distinct layers (rho = 80, 300, 50 Ohm m); ...
```

**Validity.** An empirical one-parameter description of a two-layer soil,
calibrated on the LV networks of the AP1 catalogue: $h_1$ = 5 … 30 m,
$\rho_1$ = 30 … 1000 Ω·m, $\rho_2$ = 30 … 200 Ω·m; outside that range it
is an extrapolation. The weight of the lower layer depends on the extent
of the earthing system: a compact electrode in the upper layer sees less
of it than an LV network. For the station of the
[IO example](io.md#example) (a 2 m ring and a 3 m rod at 0.8 m depth)
swept over $\rho_1$ at $\rho_2 = 100\ \Omega\,\mathrm{m}$, the standard
form fits to 13.4 / 8.3 / 3.6 % over $\rho_\text{eff}$ and to
3.9 / 1.9 / 0.6 % over $\rho_1$ for $h_1$ = 5 / 10 / 30 m. A `BusType`
still uses $\rho_\text{eff}$ to pair with its bus: check `max_relative`,
and calibrate $(w_0, \beta)$ for systems unlike the AP1 networks.
$\rho_\text{eff}$ is not `FieldResult.soil_resistivity`, which holds
$\rho_1$ of a layered soil.

## API reference

::: groundfield.soil
