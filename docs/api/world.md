# World

The :class:`groundfield.World` class is the top-level container for
the entire physics of a grounding problem: soil model, electrodes,
conductors, sources, boundary conditions and (since 0.6.0) the
ADR-0012 concrete-shell registry. It is intentionally **free of
numerics** — the numerical evaluation (backend selection, mesh
resolution, frequency list, tolerances) is configured in a
[`Engine`](solver.md) and applied to the world via
``world.solve(engine)``.

## Concrete-shell registry — cleanup contract (0.7.0)

The public :attr:`World.concrete_shell_corrections` dict accumulates
one entry per foundation electrode that was materialised with a
non-``None`` ``concrete_rho_ohm_m`` field (ADR-0012 V1 "lumped"
path). Re-using a single ``World`` across multiple
``TnNetworkGenerator.build`` calls — the canonical Monte-Carlo
pattern over the moisture distribution — would otherwise leak stale
shell resistances from earlier samples into later ones. The
explicit :meth:`World.reset_concrete_corrections` helper clears the
registry and returns a shallow copy of the dropped entries.

```python
import groundfield as gf

world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
# ... populate via TnNetworkGenerator.build(world=world) ...
dropped = world.reset_concrete_corrections()
# ... rebuild with a different moisture sample ...
```

Calling the helper on a fresh world is an idempotent no-op.

## API reference

::: groundfield.world
