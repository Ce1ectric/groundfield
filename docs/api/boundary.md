# Boundary conditions

The :mod:`groundfield.boundary` module defines the
:class:`BoundaryConditions` model that a :class:`World` carries.
The image-method, MoM and Sommerfeld backends implicitly assume a
Dirichlet far field (``φ → 0``) and a Neumann surface (``∂φ/∂n = 0``
at ``z = 0``); the explicit fields on :class:`BoundaryConditions` are
preserved for forward-compatibility with the upcoming FEM backend.

## Revert-warning contract (0.5.0)

Calling :meth:`World.set_boundary_conditions` with a non-default
value emits a :class:`UserWarning` so the user does not silently
configure a value that the v0.2.0 backends ignore. Reverting back
to the default value also emits a :class:`UserWarning` — see the
implementation note in the CHANGELOG.

## API reference

::: groundfield.boundary
