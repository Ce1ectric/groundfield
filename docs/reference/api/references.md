# References

`groundfield.references` collects the closed-form results of the
classical literature that the numerical backends are validated
against. They are independent of the solver code: every engine and
every new geometry is checked against at least one of them (see the
[validation page](../../explanation/validation.md)).

| Module | Source | Content | Used to check |
|---|---|---|---|
| `dwight1936` | Dwight, *Calculation of Resistances to Ground*, AIEE Trans., Dec. 1936, Table I | rod, two rods (close / far), horizontal wire, right-angle wire, 3/4/6/8-point star, buried ring, strip, round and vertical plate, hemisphere | grounding resistance of the image and MoM families in homogeneous soil |
| `ieee80` | IEEE Std 80-2013, Eq. 53 (Sverak; Eq. 57 of the 2000 edition) | resistance of a horizontal grid | grids; the Annex H benchmark grids are reproduced in `tests/test_ieee80_benchmark.py` |
| `oeding` | Oeding and Oswald, *Elektrische Kraftwerke und Netze* (Springer, 2016), ch. 9 | loop self and mutual inductance per unit length, internal inductance | Neumann inductance of distributed conductors |
| `carson` | Carson, Bell Syst. Tech. J. 5(4), 1926; Tleis, *Power Systems Modelling and Fault Analysis*, 2008 | worked examples and tabulated values | the Carson correction $P, Q$ |
| `earth_return` | Carson's line impedance in engineering form | self and mutual earth-return impedance per unit length, equivalent depth $D_e$, $R' = \omega\mu_0/8$ | the assembled Sommerfeld/Pollaczek stack |
| `pollaczek` | Pollaczek (1926), Sunde (1968), Tsiamitros et al. (2005) | earth-return impedance of buried conductors in two-layer soil (self, mutual, conductor–earth loop) | layered earth return |

## Mathematical and physical context

All formulas assume the same physics as the solver — quasi-static
fields, linear isotropic soil — but in idealised geometries where the
boundary-value problem has a closed or semi-closed solution. Two
examples:

- **Driven rod** of length $L$ and radius $a$, top at the surface of a
  homogeneous soil (Dwight 1936):
  $R = \dfrac{\rho}{2\pi L}\left(\ln\dfrac{4L}{a} - 1\right)$.
- **Buried conductor** at depth $d$ in two-layer soil (Pollaczek,
  Sunde): the self impedance per unit length is a spectral integral
  with the effective vertical wavenumber of the layered earth; it
  reduces to Carson's line impedance for $\rho_1 = \rho_2$ and lies
  between the two homogeneous Carson values otherwise.

Each reference states its own accuracy: Dwight's rod and ring formulas
are asymptotic in $a/L$ and accurate to well below 1 % for thin
electrodes; Sverak's grid formula is an empirical estimate that IEEE 80
quotes to a few per cent for typical substation grids.

## Example

```python
import groundfield as gf
from groundfield.references import dwight1936, ieee80


def resistance(kind, **geometry):
    """Grounding resistance in Ω of one electrode in 100 Ω·m soil."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    electrode = gf.create_electrode(world, kind, name="e", **geometry)
    gf.create_source(world, attached_to="e", magnitude=1.0)
    return gf.create_engine().solve(world).input_impedance()[0].real, electrode


R_rod, rod = resistance("rod", position=(0.0, 0.0, 0.0), length=3.0)        # 35.99 Ω
R_grid, grid = resistance(
    "grid_mesh", corner=(-10.0, -10.0, 0.5), size=(20.0, 20.0), n_x=5, n_y=5
)                                                                             # 2.31 Ω

R_dwight = dwight1936.rod(rho=100.0, length=3.0, radius=0.005)               # 35.99 Ω
R_sverak = ieee80.grid_resistance_sverak(
    rho=100.0, total_conductor_length=grid.total_length(), area=400.0, depth=0.5
)                                                                             # 2.54 Ω
assert abs(R_rod / R_dwight - 1) < 1e-3     # closed form: agreement to 0.01 %
assert abs(R_grid / R_sverak - 1) < 0.10    # empirical estimate: 9 % for this small grid
```

## API reference

::: groundfield.references.dwight1936
    options:
      heading_level: 3

::: groundfield.references.ieee80
    options:
      heading_level: 3

::: groundfield.references.oeding
    options:
      heading_level: 3

::: groundfield.references.carson
    options:
      heading_level: 3

::: groundfield.references.earth_return
    options:
      heading_level: 3

::: groundfield.references.pollaczek
    options:
      heading_level: 3
