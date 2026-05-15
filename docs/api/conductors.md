# Conductors

The ``conductors`` subpackage describes the current-carrying
elements that connect or surround electrodes but are *not*
electrodes themselves: cable shields, PEN conductors of TN
distribution networks, overhead earth wires, and auxiliary
measurement leads.

## Physical background

A conductor segment is a thin cylindrical wire of radius $a$
between two end-points $\mathbf{x}_a$ and $\mathbf{x}_b$, in or
above the soil, with material parameters
$(\rho_\text{wire}, \mu_r)$. In the frequency-domain assembly each
conductor contributes

$$
Z_b(\omega) \;=\; R \;+\; j\omega\, L_\text{Neumann}
                \;+\; \Delta Z_\text{earth-return}(\omega),
$$

with the DC resistance $R = \rho_\text{wire}\,\ell/A$, the
perfect-mirror Neumann inductance $L_\text{Neumann}$ assembled in
[Coupling](coupling.md), and the earth-return correction
($\Delta Z_\text{Carson}$ or the rigorous Sommerfeld kernel, see
ADR-0005 / ADR-0006). The conductor type tag steers the default
material parameters used by the engines:

| ``conductor_type`` | Typical use |
|---|---|
| ``"pen"`` | combined neutral / protective earth in TN systems |
| ``"cable_shield"`` | concentric cable screen, returns shield currents |
| ``"bare_copper"`` | bare-wire bonding between electrodes |
| ``"overhead"`` | overhead earth wire above ground |
| ``"generic"`` | user-supplied material parameters |

## Example

```python
import groundfield as gf

world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=500.0, h_1=1.0))

# Two ring electrodes at neighbouring houses
g1 = gf.create_electrode(world, "ring", name="g1",
                         center=(0.0, 0.0, 0.8), radius=3.0, wire_radius=0.005)
g2 = gf.create_electrode(world, "ring", name="g2",
                         center=(40.0, 0.0, 0.8), radius=3.0, wire_radius=0.005)

# PEN conductor along a 40 m cable trench between them
gf.create_conductor(
    world, name="pen_1", start=g1, end=g2,
    conductor_type="pen",
    discretize_segment_length=2.0,  # split into ~20 inductive segments
)
```

The conductor object is purely geometric / material-based.
Earth-return corrections are delegated to
:mod:`groundfield.coupling`, so the same conductor object can be
combined with the perfect-mirror approximation, with the Carson
series, or with the rigorous Sommerfeld kernel.

### Lumped series resistance (new in 0.6.0)

The optional ``lumped_series_resistance_ohm`` field overrides the
geometric series resistance $R = \rho_\text{mat}\,L / A$ with a
caller-supplied value in Ω. The :attr:`series_resistance` property
returns this value when set and the solver picks it up
transparently through the existing distributed-conductor framework
(ADR-0003).

The primary consumer is the **lumped concrete-shell path of
ADR-0012**:
:class:`~groundfield.generators.tn_network.TnNetworkGenerator`
reads each foundation's
$R_\text{shell,total} = \rho_c/(2\pi\,L_\text{perim})\,\ln(r_b/r_a)$
from ``world.concrete_shell_corrections`` (populated by the
foundation-electrode materialiser) and injects it on the PEN
service drop between the KVS and the foundation anchor. Setting
``lumped_series_resistance_ohm`` requires the conductor to be in
the finite-impedance branch — pair it with a non-``None``
``cross_section`` to ensure the override is consumed.

## API reference

::: groundfield.conductors
