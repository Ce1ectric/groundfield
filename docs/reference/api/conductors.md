# Conductors

A `Conductor` connects two nodes of the network (electrodes or
junctions) or two points: a PEN conductor of a low-voltage network, a
cable screen, a bare bonding wire, an overhead earth wire or a
measurement lead. Unlike an electrode it is not an equipotential body:
it has a series impedance, and depending on its settings it carries
inductance and leaks current into the soil along its length.

## Mathematical and physical context

A conductor of length $\ell$, wire radius $a$, material resistivity
$\rho_\text{mat}$ and cross section $A$ is a branch of the nodal
system with

$$
Z_b(\omega) = R + j\omega L_\text{Neumann} + \Delta Z_\text{earth}(\omega),
\qquad R = \rho_\text{mat}\,\ell / A ,
$$

where $L_\text{Neumann}$ (self and mutual, from the segment geometry)
and the earth-return correction $\Delta Z_\text{earth}$ (Carson or
Sommerfeld) are assembled in [Coupling](coupling.md) when the conductor
sets `inductance_model="neumann"`. The settings that decide the model:

| Field | Values | Meaning |
|---|---|---|
| `cross_section` | `None` (default) | ideal connection: both end nodes form one galvanic cluster |
| | `float` (m²) or `"from_radius"` ($\pi a^2$) | finite resistance $R = \rho_\text{mat}\ell/A$ |
| `lumped_series_resistance_ohm` | Ω | replaces the geometric $R$ |
| `discretize_segment_length` | m | distributed conductor: a chain of branches with this segment length (needs a finite resistance) |
| `coupling_to_soil` | `"isolated"` (default), `"galvanic"` | insulated cable, or bare wire that leaks along its length like an electrode |
| `inductance_model` | `None` (default), `"neumann"` | resistive branch only, or Neumann inductance with the engine's earth-return model (`Engine.earth_inductive_model`) |
| `resistance_model` | `"dc"` (default), `"skin"` | $R$ at every frequency, or the exact internal impedance of a solid round wire (Bessel functions) |

**Units.** Lengths and radii in m, `cross_section` in **m²**
(95 mm² is `95e-6`), `resistivity` in Ω·m (copper `1.68e-8`, the
default). A cross section above `MAX_PLAUSIBLE_CROSS_SECTION`
(10⁻² m²) warns with `CrossSectionUnitWarning`, because a value given in
mm² would otherwise be a silent near-ideal short.

**Skin effect.** With `resistance_model="skin"` the internal impedance
per unit length is

$$
Z'_\text{int} = \frac{k\,\rho_\text{mat}}{2\pi a}\,\frac{I_0(ka)}{I_1(ka)},
\qquad k = \sqrt{j\omega\mu_0/\rho_\text{mat}}
$$

(Schelkunoff 1934), which tends to
$\rho_\text{mat}/(\pi a^2) + j\omega\mu_0/(8\pi)$ at low frequency. Below 1 kHz the effect matters only for thick conductors:
the skin depth of copper at 1 kHz is about 2 mm.

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=1.0))
gf.create_electrode(world, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=3.0)
gf.create_electrode(world, "ring", name="g2", center=(40.0, 0.0, 0.8), radius=3.0)

pen = gf.create_conductor(
    world, name="pen_1", start="g1", end="g2",
    conductor_type="pen",
    cross_section=95e-6,             # 95 mm² in m²
    discretize_segment_length=2.0,   # distributed: about 20 inductive segments
    inductance_model="neumann",      # with the engine's earth-return model
)
assert abs(pen.length - 40.0) < 1e-12   # m, from (3, 0, 0.8) on g1 to (43, 0, 0.8) on g2
assert abs(pen.series_resistance - 1.68e-8 * 40.0 / 95e-6) < 1e-15   # 7.1 mΩ
```

The conductor object is geometry and material only; the earth-return
model is chosen on the engine (`earth_inductive_model`, default
`"sommerfeld"`), so the same world can be solved with different
earth-return models. The theory is on the
[earth return](../../explanation/earth-return.md) page.

`find_close_wires` and `find_close_conductors` report pairs of wires
that are closer than the thin-wire model allows (interpenetrating,
crossing or very close); [`diagnose`](diagnostics.md) runs them before a
solve.

## API reference

::: groundfield.conductors
