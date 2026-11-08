# Boundary conditions

`BoundaryConditions` records the boundary conditions of the field
problem on the `World`. Every backend implements the same set — remote
earth as the potential reference and an insulating soil surface — so
the fields exist for documentation and forward compatibility only. A
non-default value is stored but read by no backend and warns with
`InertConfigurationWarning`.

## Mathematical and physical context

Below 1 kHz the displacement current in the soil is negligible, the
leakage current density follows Ohm's law
$\mathbf{J} = -\sigma\nabla\varphi$, and charge conservation
$\nabla\cdot\mathbf{J} = 0$ gives

$$
\nabla\cdot\bigl(\sigma(\mathbf{r})\,\nabla\varphi(\mathbf{r})\bigr) = 0 ,
$$

Laplace's equation $\nabla^2\varphi = 0$ inside every homogeneous
layer. The injected currents are not a volume source; they enter
through the electrode surfaces. A well-posed problem needs three
conditions, which are the three fields of this model:

| Field | Value used by every backend | Meaning |
|---|---|---|
| `far_field` | `"dirichlet"` | $\varphi \to 0$ for $\lvert\mathbf{r}\rvert \to \infty$: remote earth is the reference of every potential |
| `surface` | `"neumann"` | $\partial\varphi/\partial z = 0$ at $z = 0$: no current into the air; realised by the mirror image at $z \to -z$ |
| `reference_node` | `None` | the datum is remote earth |

At an interface between layers $\varphi$ and the normal current density
$\sigma\,\partial_n\varphi$ are continuous; the layered Green's
functions of the solvers satisfy these conditions, not this model.

**Validity.** Linear, isotropic, piecewise homogeneous soil in the
quasi-static regime. The inductive coupling of conductors (and its
earth-return correction) is a separate branch impedance of the
conductor network, not part of this electrostatic-type boundary-value
problem.

## Example

```python
import warnings

import groundfield as gf
from groundfield.utils import InertConfigurationWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    boundary = gf.BoundaryConditions(surface="dirichlet")   # stored, not used
assert [w.category for w in caught] == [InertConfigurationWarning]
world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0), boundary=boundary)
```

`World.set_boundary_conditions(...)` stores a changed value as well and
warns with a plain `UserWarning`, also when a value is set back to the
default.

## API reference

::: groundfield.boundary
