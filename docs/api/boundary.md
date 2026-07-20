# Boundary conditions

The :mod:`groundfield.boundary` module defines the
:class:`BoundaryConditions` model that a :class:`World` carries.
The image-method, MoM and Sommerfeld backends implicitly assume a
Dirichlet far field (``φ → 0``) and a Neumann surface (``∂φ/∂n = 0``
at ``z = 0``); the explicit fields on :class:`BoundaryConditions` are
preserved for forward-compatibility with the upcoming FEM backend.

## Mathematical / physical model

groundfield solves the **quasi-static** grounding problem. Below roughly
1 kHz the displacement current in the soil is negligible, so the leakage
current density follows Ohm's law $\vec{J} = \sigma\,\vec{E} =
-\sigma\,\nabla\varphi$ and charge conservation $\nabla\cdot\vec{J} = 0$ in
the source-free bulk reduces the potential to **Laplace's equation**

$$
\nabla\cdot\!\bigl(\sigma(\vec{r})\,\nabla\varphi(\vec{r})\bigr) = 0,
$$

which becomes $\nabla^2\varphi = 0$ inside every homogeneous soil region.
Injected electrode currents are **not** a volume source term; they enter
through the electrode boundary (the solver's right-hand side), which is why
they are not part of :class:`BoundaryConditions`.

A well-posed boundary-value problem needs three conditions, and these are
exactly the fields this module carries:

- **Far field (`far_field`, remote earth).** Dirichlet $\varphi \to 0$ as
  $|\vec{r}| \to \infty$. The integral / image-charge backends realise it
  through the $1/r$ decay of the free-space Green's function, so all
  reported potentials are referred to remote earth.
- **Soil surface (`surface`, air–soil interface at $z = 0$).** Homogeneous
  Neumann $\partial\varphi/\partial n = \partial\varphi/\partial z = 0$: no
  leakage current crosses into the insulating air. The backends realise it
  with a mirror image charge at $z \to -z$, which doubles the potential of a
  buried source in the lower half-space.
- **Potential datum (`reference_node`).** Fixes the additive gauge of
  $\varphi$. With the default Dirichlet far field the datum is remote earth;
  an explicit reference node is reserved for the FEM backend.

At an interface between two soil layers of conductivity $\sigma_k$ and
$\sigma_{k+1}$ the potential $\varphi$ and the normal current density
$J_n = \sigma\,\partial_n\varphi$ are continuous; those interface conditions
are handled by the layered Green's functions in the
[coupling](coupling.md) package, not by this module.

**Validity.** Linear, isotropic, piecewise-homogeneous soil in the
quasi-static regime. Frequency-dependent earth-return (inductive) effects
are treated as a separate additive correction on the branch impedance (see
[coupling](coupling.md)), not as part of this electrostatic boundary-value
problem.

## Revert-warning contract (0.5.0)

Calling :meth:`World.set_boundary_conditions` with a non-default
value emits a :class:`UserWarning` so the user does not silently
configure a value that the v0.2.0 backends ignore. Reverting back
to the default value also emits a :class:`UserWarning` — see the
implementation note in the CHANGELOG.

## API reference

::: groundfield.boundary
