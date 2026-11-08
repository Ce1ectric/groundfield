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

with $h_\text{ref} = 10$ m and weights $(w_0, \beta)$ of the electrode
class ([presets](#presets-per-electrode-class)). The default is the AP1
calibration over all LV networks, $w_0 = 0.13$, $\beta = 0.8$: the
weight of the lower layer falls with the thickness of the upper layer,
$w$ = 0.47, 0.23, 0.13 and 0.054 for $h_1$ = 2, 5, 10 and 30 m. AP2
gives a bus $\rho_\text{eff}$, so the `rho` of a `BusType` fitted in
layered soil is $\rho_\text{eff}$ of the soil it was fitted in
([IO and groundinsight](io.md)).

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
  $h_1 < h_\text{ref}\,w_0^{1/\beta}$ (0.78 m with the default weights,
  0.28 … 1.34 m over the presets) and $\rho_\text{eff}$ leaves the
  interval (17.9 Ω·m for $\rho_1 = 100$, $\rho_2 = 30\ \Omega\,\mathrm{m}$,
  $h_1 = 0.5$ m); with the limit such a thin upper layer gives $\rho_2$;
- is bit-identical, inside the calibration range, to
  `r1 ** (1 - w) * r2 ** w` of the AP1 / AP2 scripts with the same
  weights.

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

### Presets per electrode class

The weights depend on the extent of the earthing system: a compact
electrode in the upper layer sees less of the lower layer than an LV
network with its foundation electrodes. `gf.soil.EFFECTIVE_RESISTIVITY_PRESETS`
is a read-only mapping from a name to the weights `(w0, beta)` of one
electrode class, all at $h_\text{ref} = 10$ m;
`gf.effective_resistivity(soil, preset=...)` uses them, and an explicit
`w0` or `beta` replaces that weight only.

| Preset | Electrode class | $w_0$ | $\beta$ |
|---|---|---|---|
| `"lv_network"` (default) | all LV networks of the AP1 catalogue (global calibration, the hand-over's method A0) | 0.13 | 0.8 |
| `"station"` | the station electrode alone: 4 m × 4 m ring with three 4 m rods, no foundation electrodes, purely resistive | 0.06 | 1.3 |
| `"lv_group_<g>_p<pp>"` | LV network of group $g$ with $pp$ % foundation electrodes | table below | |

$(w_0, \beta)$ of `"lv_group_<g>_p<pp>"`, fitted per group and
penetration (method A, the values of the group's median network, q50):

| Group | p = 5 % | p = 10 % | p = 20 % | p = 40 % |
|---|---|---|---|---|
| 1 single objects, hamlets | 0.06, 1.1 | 0.04, 1.6 | 0.04, 1.6 | 0.06, 1.3 |
| 2 villages | 0.06, 1.4 | 0.10, 0.9 | 0.16, 0.9 | 0.24, 0.5 |
| 3 large villages, districts | 0.12, 0.9 | 0.16, 0.6 | 0.20, 0.5 | 0.24, 0.4 |
| 4 dense built-up areas | 0.12, 0.8 | 0.18, 0.6 | 0.28, 0.4 | 0.36, 0.3 |

An LV network belongs to a group by its perimeter equivalent
$n_U = \sum_i U_i / 40\ \mathrm{m}$ ($U_i$ the perimeter of building
$i$) with the thresholds 20, 100 and 250, so `"lv_group_2_p10"` is a
village with 10 % foundation electrodes.

**Provenance.** The values are those of the author's AP2 hand-over
(`gruppen_bustypes.json` and `bericht_gruppen.md`, generated on
2026-10-20 by `gruppenbildung.py` from the AP1 catalogue computed with
groundfield 0.14.0). Their calibration range is that of the catalogue:
$h_1$ = 5 … 30 m, $\rho_1$ = 30 … 1000 Ω·m, $\rho_2$ = 30 … 200 Ω·m. They
belong to that catalogue and are re-derived when it is recomputed (the
author re-runs it with groundfield 2026.11). The hand-over's q10 and q90
BusTypes carry weights of their own, which are not presets.

```python
presets = gf.soil.EFFECTIVE_RESISTIVITY_PRESETS
assert presets["lv_network"] == (0.13, 0.8) and presets["station"] == (0.06, 1.3)

soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=5.0)
network = gf.effective_resistivity(soil)                         # "lv_network": w = 0.226
station = gf.effective_resistivity(soil, preset="station")       # w = 0.06 * 0.5**-1.3 = 0.148
village = gf.effective_resistivity(soil, preset="lv_group_2_p10")  # w = 0.10 * 0.5**-0.9 = 0.187
assert abs(network - 76.1465) < 1e-4 and abs(station - 83.7050) < 1e-4 and abs(village - 79.8780) < 1e-4
assert gf.effective_resistivity(soil, preset="station", beta=0.8) == gf.effective_resistivity(soil, w0=0.06)

try:
    gf.effective_resistivity(soil, preset="statoin")
except ValueError as error:
    print(error)    # effective_resistivity: unknown preset 'statoin'. Did you mean 'station'? ...
```

A `BusType` and the bus it is paired with must use the same weights:
fit the `BusType` of a station over $\rho_\text{eff}$ with `"station"`
and give its bus the same $\rho_\text{eff}$; the BusType metadata
records only that `rho` is an effective resistivity, so name the preset
in `soil_summary` ([IO and groundinsight](io.md)).

**Validity.** An empirical one-parameter description of a two-layer
soil, calibrated on the AP1 catalogue; outside the calibration range it
is an extrapolation, and a preset belongs to its electrode class. For
the AP1 station (the ring and rods above, at 0.6 m) solved in the 39
soils of the AP1 grid, $R = c\,\rho_\text{eff}$ with $c$ from
homogeneous soil misses the field solution by a median 2.1 % and at most
15.7 % with `"station"` (the hand-over reports 1.7 % and 15.7 %), by
5.5 % and 35.9 % with `"lv_network"` and by 12.4 % and 41.9 % over
$\rho_1$ (rms 5.5, 11.8 and 18.9 %); the LV network presets give an rms
error of 5.5 % (group 1) up to 39.9 % (group 4, 40 %) there. For the station
of the [IO example](io.md#example) (a 2 m ring and a 3 m rod at 0.8 m)
swept over $\rho_1$ at $\rho_2 = 100\ \Omega\,\mathrm{m}$, the standard
form fits to 7.5 / 2.9 / 0.6 % over $\rho_\text{eff}$ with `"station"`,
13.4 / 8.3 / 3.6 % with `"lv_network"` and 3.9 / 1.9 / 0.6 % over
$\rho_1$ for $h_1$ = 5 / 10 / 30 m: at fixed $\rho_2$ the constant $k_0$
absorbs the lower layer, which favours $\rho_1$, but a bus carries
$\rho_\text{eff}$. Check `max_relative`, and calibrate $(w_0, \beta)$
for systems unlike the hand-over's classes. $\rho_\text{eff}$ is not
`FieldResult.soil_resistivity`, which holds $\rho_1$ of a layered soil.

## API reference

::: groundfield.soil
