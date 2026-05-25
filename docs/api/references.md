# Analytical reference formulas

The :mod:`groundfield.references` subpackage contains analytical
reference formulas used in plausibility tests and as closed-form
sanity checks against the numerical backends.

## Dwight 1936 — straight rod in homogeneous soil

[`groundfield.references.dwight1936`](dwight1936) implements the
classical formula for the spreading resistance of a single vertical
rod electrode in homogeneous soil:

$$
R = \frac{\rho}{2\pi L}\,\left[\ln\!\frac{4 L}{a} - 1\right]
$$

with $\rho$ the soil resistivity, $L$ the rod length and $a$ the
rod radius. The formula is exercised by
``tests/test_dwight_references.py`` against the image-method backend.

## Carson 1926 — overhead-line earth-return impedance

[`groundfield.references.carson`](carson) implements Carson's
series expansion for the earth-return self/mutual impedance of an
overhead conductor over a homogeneous lossy half-space.

## Oeding — inductive coupling between buried conductors

[`groundfield.references.oeding`](oeding) implements the
closed-form expressions for the mutual inductance between buried
conductors in homogeneous soil from Oeding & Oswald,
*Elektrische Kraftwerke und Netze*.

## API reference

::: groundfield.references
