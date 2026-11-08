# Postprocess

The `postprocess` package turns a `FieldResult` into engineering
quantities: potentials and their plots, touch and step voltages with the
EN 50522 limits, the current sharing between electrodes and conductors,
parameter sweeps and convergence studies, and the reduced `rho-f` models
that are exported to `groundinsight`. Every function is available at the
top level (`gf.touch_voltage`) and in the namespaces `gf.plot`,
`gf.safety` and `gf.fit`.

## Mathematical and physical context

**Potentials.** After the solve the leakage currents $I_k$ of the
segments are known, and the potential anywhere in or on the soil is the
superposition $\varphi(\mathbf{r}) = \sum_k G(\mathbf{r}, \mathbf{r}_k)\,I_k$
with the Green's function of the soil (homogeneous or two-layer).

**Touch and step voltage** (EN 50522:2022, IEC 61936-1). A person
touches an accessible conductive part at the potential
$U_\text{part}$ of the earthing system it is bonded to and stands on the
surface 1 m away from it (EN 50522's conventional position):

$$
U_T = U_\text{part} - \varphi(\mathbf{r}_\text{feet}),
\qquad
U_S = \varphi(\mathbf{r}_1) - \varphi(\mathbf{r}_1 + d_\text{step}\,\hat{\mathbf{e}}).
$$

`touch_voltage_at_part` evaluates $U_T$ at a touch point, an accessible
metal part above ground (a station housing or door, a mast, a fence
post, a service connection box): the feet stand on the circle of radius
$d$ (default 1 m) around the part, optionally only in given directions
or outside the outlines where nobody can stand, and $U_\text{part}$ is
the potential of the electrode or junction the part is bonded to — a
PEN junction at a house gives the transferred potential. Where the
parts are not known, `touch_voltage` and `touch_voltage_envelope` take
the electrode itself as the touched part, $U_\text{part} = U_E =
\varphi_\text{cluster}$, with the feet at distance $d$ from its
footprint, the horizontal projection of its wires; the envelope walks
all around it. It samples nothing inside the footprint by default (there
stands the station, the house, the steel structure or the mast); for
open-air grids such as a substation yard, `interior="meshes"` adds the
centres of the meshes (the analogue of the IEEE 80 mesh voltage) and of
a ring, `interior=True` also the inside of closed polylines.
`permissible_touch_voltage_en50522(t_f)` follows Figure 8 / Table B.4 of
EN 50522:2022 up to 10 s and returns 80 V for every $t_f > 10$ s,
including an uncleared fault (`math.inf`), the value the standard allows
for current-flow durations considerably longer than 10 s;
`permissible_prospective_touch_voltage_en50522` adds the body and
footwear resistances of Annex B.4. For a fault current with harmonics,
`touch_voltage_spectrum` returns the true-RMS value
$U_{T,\text{rms}} = \sqrt{\sum_k \lvert U_{T,k}\rvert^2}$ over the
frequencies of the spectrum.

**Current sharing.** `cluster_current_balance` and
`electrode_current_table` list the current every cluster and electrode
leaks into the soil; `split_factor` is the share of a source current
that leaves through the soil at the feeding cluster (the rest flows
away through conductors). The inductive reduction factor of cable
screens is not computed here.

**Reduced models.** The quantity a reduced model describes is selected
with `quantity=` (`"input"`, the default, `"cluster"` or `"electrode"`,
see the [quickstart](../../getting-started/quickstart.md)). Two families:

- *vector fitting* (Gustavsen and Semlyen 1999) of a rational function
  in $s = j\omega$ at a fixed soil,
  $Z(s) \approx R_\infty + sL_\infty + \sum_k r_k/(s - p_k)$, with a
  passivity check ($\operatorname{Re}Z \ge 0$) before export;
- the *standard form* fitted over a $\rho$ and $f$ sweep,

  $$
  Z(\rho, f) = k_0 + k_1\rho + (k_2 + jk_3)\,f + (k_4 + jk_5)\,f\rho ,
  $$

  with relative weights by default; `max_relative` is the quality figure
  to check (a fit above 5 % warns with `RhoFStandardQualityWarning`).

**Validity.** Quasi-static range $f < 1\,\mathrm{kHz}$. The image family
assumes a uniform current per unit length on each electrode: near grids,
meshes, stars, strip ends and bonded multi-electrode clusters its surface
potentials are impedance-grade, not field-grade, and the safety helpers
warn with `UniformCurrentFieldWarning` (re-solve with `backend="mom"` for
field-grade values). `FieldResult.potential` and everything built on it
support homogeneous and two-layer soil only.

## Example

A ring earth carrying a fault current of 100 A at 50 Hz, the touch
voltage 1 m outside its footprint, checked against a second backend and
against the EN 50522 limit for a fault cleared in 0.5 s:

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=2.0))
gf.create_electrode(world, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=3.0)
gf.create_source(world, attached_to="g1", magnitude=100.0)
result = gf.create_engine(frequencies=[50.0]).solve(world)

U_E = result.electrode_potentials["g1"][0]                       # 1381 V
U_T = gf.touch_voltage(result, world, electrode="g1", distance=1.0)   # 525 V
U_S = gf.step_voltage(result, position=(3.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), world=world)
U_Tp = gf.permissible_touch_voltage_en50522(0.5)                  # 225 V (Table B.4)

mom = gf.create_engine(backend="mom", frequencies=[50.0]).solve(world)
U_T_mom = gf.touch_voltage(mom, world, electrode="g1", distance=1.0)  # 517 V

assert abs(U_E - 1380.9) < 0.5 and abs(U_T - 525.3) < 0.5
assert abs(U_T - U_T_mom) / abs(U_T) < 0.02     # ring: both current ansätze agree
assert U_Tp == 225.0 and abs(U_T) > U_Tp        # this earth needs a closer look
assert 0 < U_S.real < U_T.real
```

The how-to guide [Assess touch voltages](../../how-to/assess-touch-voltages.md)
and tutorial T08 show the complete assessment (touch points, transferred
potential, envelope, prospective touch voltage, spectrum); tutorial T04
the frequency response and the `rho-f` fits.

## Safety (touch and step voltage)

::: groundfield.postprocess.safety
    options:
      heading_level: 3

## Current sharing

::: groundfield.postprocess.current_balance
    options:
      heading_level: 3

## Sweeps and convergence studies

::: groundfield.postprocess.sweep
    options:
      heading_level: 3

::: groundfield.postprocess.convergence
    options:
      heading_level: 3

## Vector fitting

::: groundfield.postprocess.vector_fitting
    options:
      heading_level: 3

## Standard rho-f form

::: groundfield.postprocess.rho_f_standard
    options:
      heading_level: 3

## Potential plots

::: groundfield.postprocess.plotting
    options:
      heading_level: 3

## Geometry plots

::: groundfield.postprocess.geometry_plot
    options:
      heading_level: 3
