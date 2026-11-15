# Backend registry

Every numerical backend of `Engine` is a registered backend
(`groundfield.solver.registry`, ADR-0014). A registration ties a name to
its solve function, its declared **capabilities**, an optional
**options model** for `Engine.options`, and flags for deprecated aliases,
experimental status and the automatic forwarding by soil model.
`Engine.solve` resolves the name, applies the forwarding, checks the
capabilities against the world *before* any backend code runs, calls the
solve function and fills the common metadata of the result.

## Built-in backends

| Name | Soils | Kernels | Concrete shells | Two-terminal sources | Junctions | Voltage sources | Status |
|---|---|---|---|---|---|---|---|
| `image` | homogeneous (forwards two- and multi-layer) | line, point | yes | yes | yes | yes | default |
| `image_2layer` | two-layer | line, point | yes | yes | yes | yes | |
| `mom` | homogeneous, two-layer | point | yes (since 2026.11) | yes | yes | yes | |
| `mom_sommerfeld` | any number of layers | point | yes (since 2026.11) | yes | yes | yes | options: `MomSommerfeldOptions` |
| `fem` | any number of layers | — | yes (since 2026.11) | yes (since 2026.11) | yes (since 2026.11) | yes (since 2026.11) | experimental, axisymmetric worlds only, options: `FemOptions` |
| `cim` | → `image` | | | | | | deprecated alias |
| `bem` | → `mom` | | | | | | deprecated alias |
| `image_nlayer` | homogeneous, two-layer | line, point | yes | yes | yes | yes | deprecated name of the `image` dispatch |

Every built-in backend honours every conductor model (lumped,
distributed, inductive, skin effect). Since its conforming model of
2026.11 (user decision Q37) `fem` runs the network of the shared pipeline
like the integral backends; it refuses worlds that are not axisymmetric
(rods on one vertical axis, horizontal rings centred on it) and galvanic
distributed conductors, whose leakage runs along their route. Up to then
it solved distributed and inductive conductors as lumped resistive
branches and rejected shells, junctions, two-terminal and voltage
sources. The table is checked against the registry:

```python
import groundfield as gf
from groundfield.solver import get_backend

assert gf.available_backends() == ("image", "image_2layer", "mom", "mom_sommerfeld", "fem")
assert set(gf.available_backends(include_deprecated=True)) - set(gf.available_backends()) == {
    "cim", "bem", "image_nlayer"
}

caps = {name: get_backend(name).capabilities for name in gf.available_backends()}
assert caps["image"].kernels == {"line", "point"} and caps["mom"].kernels == {"point"}
assert caps["mom_sommerfeld"].soils == {"homogeneous", "two_layer", "multi_layer"}
assert caps["image"].concrete_shells and caps["mom"].concrete_shells
assert caps["mom_sommerfeld"].concrete_shells and caps["fem"].concrete_shells
assert caps["fem"].two_terminal_sources and caps["fem"].junctions
assert caps["fem"].conductors == {"lumped", "distributed", "inductive", "skin"}
assert all(caps[n].voltage_sources for n in gf.available_backends())
assert get_backend("fem").experimental
assert get_backend("cim").deprecated_alias_of == "image"
assert get_backend("bem").deprecated_alias_of == "mom"
```

## What the capability check does

A world that a backend **cannot represent** raises
`BackendCapabilityError` (a subclass of `ValueError` and
`NotImplementedError`) naming the backends that can: an unsupported
layer count, electrodes with a concrete-shell coefficient, junction
nodes, voltage sources, or an explicitly requested kernel the backend
does not have.
Adjacent layers of equal resistivity count as one layer.

A world that a backend **solves with a documented simplification**
warns with `BackendCapabilityWarning` and solves: conductor models the
backend reduces (no built-in backend since 2026.11), or a `return_to`
terminal on a backend that returns the current through remote earth.

Solving with a deprecated alias warns with `DeprecationWarning`; the
first solve with an experimental backend in a process warns once with
`ExperimentalBackendWarning`. Options for a backend that will not run
warn with `BackendOptionsWarning`.

## Registering a backend

`register_backend(name, solve_fn, capabilities=..., options_model=...)`
adds a backend without changing `Engine.solve`; the solve function has
the signature `solve_fn(world, engine, **options) -> FieldResult`. A
registered name is accepted by `Engine(backend=...)` and listed by
`available_backends()`.

## API reference

::: groundfield.solver.registry
