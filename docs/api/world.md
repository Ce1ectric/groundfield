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

## Source handles stay live across a solve (0.15.0)

``World.solve(engine)`` snapshots every :attr:`World.sources` entry
before the backend runs and restores it afterwards, so solving never
rewrites the input world (opt out with ``snapshot_sources=False`` if a
sweep cannot afford the deep copy). Since 0.15.0 that roll-back is
**identity-preserving**: the recorded field values are written back onto
the very objects ``create_source`` returned, and ``world.sources`` keeps
its list object. The canonical parameter sweep therefore behaves as it
reads:

```python
import groundfield as gf

world  = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
src    = gf.create_source(world, attached_to="g1", magnitude=10.0)
engine = gf.create_engine(backend="image")

for current in (10.0, 20.0, 40.0):
    src.magnitude = current       # ``src`` is still world.sources[0]
    print(world.solve(engine).electrode_potentials["g1"][0])
```

Up to 0.14.x the snapshot list was rebound onto the world, which
detached the caller's handle after the first solve: every later
mutation of ``src`` was a silent no-op and the loop above printed the
same potential three times. Backend-side mutations of the source list
(insert / remove / replace / wholesale rebind) are still rolled back.

## API reference

::: groundfield.world
