"""The PEN cable of a generated TN network lies at ``PenConfig.depth_m``.

Up to v0.15.0 (and in the first 2026.11 consolidation) ``PenConfig.depth_m``
was stored but never read: every PEN section ran straight between the
connection points of its end electrodes, e.g. from the substation ring at
0.6 m to the head of a cabinet rod at 0 m. Since 2026.11 every PEN section
and service drop runs horizontally at ``z = depth_m`` between the
horizontal positions of its end anchors (the cable trench); the short
risers to the electrode heads are not modelled, as for the measurement
leads (F48). Same class of defect as F48 (review finding 10.55, an inert
depth field), observed by the GEN-1 stream.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
from groundfield.generators import (
    HomogeneousSoilSpec,
    ManhattanGridPlacement,
    PenConfig,
    RadialTrunkTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)

pytestmark = pytest.mark.finding("10.55")


def _world(depth_m: float = 0.6, **kw: object) -> gf.World:
    # The houses stand on a row 20 m beside the main: since user decision
    # Q41 the default row on the main is joined to it without drops.
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": 4},
        placement=ManhattanGridPlacement(
            spacing_x_m=25.0, spacing_y_m=30.0, n_per_row=10, centre_xy=(0.0, 20.0)
        ),
        pen=PenConfig(depth_m=depth_m, **kw),
    )
    return TnNetworkGenerator(cfg, seed=0).build()


def _pen(world: gf.World) -> list[gf.Conductor]:
    return [c for c in world.conductors if c.conductor_type == "pen"]


def _node_point(world: gf.World, name: str) -> tuple[float, float, float]:
    """Connection point of an electrode or a junction (PEN tap point)."""
    if any(j.name == name for j in world.junctions):
        return world.get_junction(name).connection_point
    return world.get_electrode(name).connection_point


@pytest.mark.parametrize("depth", [0.6, 1.2])
def test_pen_sections_run_at_the_configured_depth(depth: float) -> None:
    world = _world(depth)
    sections = _pen(world)
    # The main is split at the four tap points of the service drops.
    drops = [c for c in sections if c.name.startswith("pen_service_")]
    assert len(drops) == 4 and len(sections) == 4 + 5
    for c in sections:
        assert c.start[2] == c.end[2] == depth
        for name, point in ((c.start_electrode, c.start), (c.end_electrode, c.end)):
            if "_foundation_" in name:
                # User decision Q41: the point of the wire nearest to the
                # conductor, in plan, at the PEN depth.
                (run,) = world.get_electrode(name).wire_polylines()
                (x0, y0, _), (x1, y1, _) = run[0], run[-1]
                assert (
                    x0 == x1 == point[0] and min(y0, y1) <= point[1] <= max(y0, y1)
                ) or (y0 == y1 == point[1] and min(x0, x1) <= point[0] <= max(x0, x1))
            else:
                assert point[:2] == _node_point(world, name)[:2]


def test_radial_trunk_sections_run_at_the_configured_depth() -> None:
    cfg = TnNetworkConfig(
        building_counts={"residential": 12},
        pen_topology=RadialTrunkTopology(slots_per_substation=3),
        pen=PenConfig(depth_m=0.9),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    sections = _pen(world)
    assert any(c.name.startswith("pen_trunk_") for c in sections)
    assert all(c.start[2] == c.end[2] == 0.9 for c in sections)


def test_depth_moves_the_galvanic_pen_result() -> None:
    """A bare (galvanic) PEN leaks along its length: its depth matters."""
    engine = gf.create_engine(backend="image", segment_length=1.0, frequencies=[50.0])
    z = {}
    for d in (0.8, 1.6):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = engine.solve(_world(d, coupling_to_soil="galvanic"))
        z[d] = res.input_impedance()[0]
    assert abs(z[0.8] - z[1.6]) > 1e-3 * abs(z[0.8])


def test_depth_must_be_below_the_surface() -> None:
    with pytest.raises(ValueError):
        PenConfig(depth_m=0.0)
    with pytest.raises(ValueError):
        PenConfig(depth_m=-0.5)
