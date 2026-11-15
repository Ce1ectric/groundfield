"""Worlds that mix projected and local coordinates are flagged (user decision D5).

Audit W2B-04: a world with its electrodes at UTM coordinates (easting
32 604 321 m with the zone prefix) and one node or electrode given in local
coordinates near (0, 0) is solved as given. Its parts lie 33 000 km apart
and a conductor between them is that long; nothing warned, ``World.check()``
passed and ``diagnose()`` reported "no findings". Measured by W2B (evidence
``b2_mixed_worlds.py``, re-run by stream fw2mom; a 12 m x 7 m outline and
a 3 m rod at UTM coordinates): a return rod at local (80, 15) m raises Z_in
by 0.43 % (the potential next to the outline by 5.8 %; the ring of the
tests below +0.41 %); a PEN cable to a cabinet at local coordinates becomes a conductor
of 33 118 097 m, Z_in 7494 + j25396 Ohm instead of 5.39 + j0.19 Ohm with
Carson's earth return, and the default Sommerfeld earth return does not
finish. The round-off of such a world is harmless (1.5e-11 against
1.0e-11 of the pure UTM world, point kernel).

Since 2026.11:

* ``diagnose()`` reports the category ``coordinates``: a *warning* when
  points lie both within 10 km of the origin and beyond 1000 km from it,
  an *info* when the horizontal extent of the world (the diagonal of its
  bounding box) exceeds 100 km;
* the solve of such a mixed world (image family, ``mom``,
  ``mom_sommerfeld``, ``solve_mutual_*``) warns once with
  :class:`groundfield.diagnostics.MixedCoordinatesWarning`, before any
  assembly.

A pure UTM world and a pure local world, however large within 100 km, are
neither reported nor warned about.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import groundfield as gf
from groundfield.diagnostics import MixedCoordinatesWarning, diagnose

UTM = (32_604_321.123, 5_812_345.678)
HOM = gf.HomogeneousSoil(rho=100.0)
TWO = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=3.0)


def _mixed_world():
    """W2B's world: a ring at UTM, its return rod given at local (80, 15) m."""
    w = gf.create_world(soil=HOM)
    gf.create_electrode(w, "ring", name="h", center=(UTM[0], UTM[1], 0.8), radius=5.0)
    gf.create_electrode(w, "rod", name="aux", position=(80.0, 15.0, 0.0), length=1.0)
    gf.create_source(w, attached_to="h", return_to="aux", magnitude=1.0)
    return w


def _pen_to_local_cabinet():
    """W2B's second world: a PEN cable from a UTM rod to a cabinet at local coordinates."""
    w = gf.create_world(soil=HOM)
    gf.create_electrode(w, "rod", name="h", position=(UTM[0], UTM[1], 0.7), length=3.0)
    gf.create_junction(w, name="cabinet", position=(250.0, 40.0, 0.7))
    gf.create_conductor(
        w,
        name="pen",
        start="h",
        end="cabinet",
        cross_section=95e-6,
        inductance_model="neumann",
        wire_radius=0.006,
    )
    gf.create_source(w, attached_to="cabinet", magnitude=1.0)
    return w


def _two_rods(a, b, soil=HOM):
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="a", position=(a[0], a[1], 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="b", position=(b[0], b[1], 0.5), length=3.0)
    gf.create_source(w, attached_to="a", return_to="b", magnitude=1.0)
    return w


def _mixed_records(rec) -> list:
    return [r for r in rec if issubclass(r.category, MixedCoordinatesWarning)]


def _solve_recording(world, engine):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = engine.solve(world)
    return result, _mixed_records(rec)


# ---------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------


# audit W2B-04; user decision D5
def test_w2b_04_a_world_mixing_projected_and_local_coordinates_is_flagged():
    """W2B-04 (D5): a return electrode given in local coordinates is 33 000 km away.

    The solve succeeded without any message (Z_in +0.41 % for this ring,
    +0.43 % for W2B's outline world), World.check() passed and diagnose()
    reported "no findings".
    """
    report = diagnose(_mixed_world(), gf.create_engine(segment_length=0.5))
    text = str(report).lower()
    assert "no findings" not in text
    assert any(word in text for word in ("coordinate", "extent", "span", "km"))
    (finding,) = report.by_category("coordinates")
    assert finding.severity == "warning"
    assert finding.data["near"] == ["aux"]
    assert finding.data["far"] == ["h"]
    assert finding.data["spanning"] == []
    # The two parts lie 33 118 km apart (the diagonal of the bounding box of
    # the ring centre and the rod head).
    assert finding.data["extent_m"] == pytest.approx(math.dist(UTM, (80.0, 15.0)))
    assert "33 118 km" in finding.message
    # A warning-severity finding is part of report.warn().
    with pytest.warns(gf.DiagnosticsWarning, match="coordinates"):
        report.warn()


# audit W2B-04; user decision D5
def test_the_finding_names_the_conductor_between_the_two_groups():
    """The PEN to a cabinet at local coordinates is a 33 118 km conductor."""
    report = diagnose(_pen_to_local_cabinet())  # geometry only, no engine
    (finding,) = report.by_category("coordinates")
    assert finding.severity == "warning"
    assert finding.data["near"] == ["cabinet"]
    assert finding.data["far"] == ["h"]
    assert finding.data["spanning"] == ["pen"]
    length = _pen_to_local_cabinet().conductors[0].length
    assert length > 3.3e7
    assert finding.data["spanning_lengths_m"] == [pytest.approx(length)]
    assert "'pen'" in finding.message


# audit W2B-04; user decision D5
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((-75_000.0, 0.0), (75_000.0, 0.0)),  # local, 150 km
        (UTM, (UTM[0] + 800_000.0, UTM[1])),  # two UTM zones (prefix 32 / 33)
    ],
    ids=["local_150km", "two_utm_zones"],
)
def test_a_world_wider_than_100_km_is_information(a, b):
    """An extent above 100 km without mixing is an info finding, and no warning."""
    world = _two_rods(a, b)
    report = diagnose(world, gf.create_engine(segment_length=0.5))
    (finding,) = report.by_category("coordinates")
    assert finding.severity == "info"
    assert finding.data["extent_m"] == pytest.approx(math.dist(a, b))
    assert finding.data["extent_m"] > 1e5
    # Not part of report.warn() (warnings only).
    assert all(
        "coordinates" not in line for line in report.lines(min_severity="warning")
    )
    _, mixed = _solve_recording(world, gf.create_engine(segment_length=0.5))
    assert mixed == []


def test_elements_are_classified_by_their_own_points():
    """Self-attack of stream fw2mom: names may repeat across kinds.

    A conductor may carry the name of an electrode; each element is
    classified by its own points, so the far electrode is not taken for a
    conductor between the groups. A single polyline with local and projected
    vertices mixes on its own.
    """
    from groundfield.solver._frame import coordinate_span

    world = gf.create_world(soil=HOM)
    gf.create_electrode(
        world, "rod", name="h", position=(UTM[0], UTM[1], 0.5), length=1.0
    )
    gf.create_electrode(world, "rod", name="x", position=(10.0, 0.0, 0.5), length=1.0)
    gf.create_electrode(world, "rod", name="y", position=(20.0, 0.0, 0.5), length=1.0)
    gf.create_conductor(world, name="h", start="x", end="y")
    span = coordinate_span(world)
    assert (span.near, span.far, span.spanning) == (("x", "y", "h"), ("h",), ())
    assert span.mixed

    line = gf.create_world(soil=HOM)
    gf.create_electrode(
        line,
        "polyline",
        name="p",
        closed=False,
        vertices=[(5.0, 0.0, 0.5), (UTM[0], UTM[1], 0.5)],
    )
    span = coordinate_span(line)
    assert (span.near, span.far, span.spanning) == ((), (), ("p",))
    assert span.spanning_lengths_m == (None,)
    assert span.mixed
    (finding,) = diagnose(line).by_category("coordinates")
    assert finding.severity == "warning" and "'p'" in finding.message


@pytest.mark.parametrize(
    ("near", "far", "flagged"),
    [
        (1.0e4, 1.0e6 + 1.0, True),  # inclusive at 10 km, exclusive at 1000 km
        (1.0e4 + 1.0, 1.0e6 + 1.0, False),
        (1.0e4, 1.0e6, False),
    ],
)
def test_the_two_radii_of_the_mixed_check(near, far, flagged):
    """Within 10 km (inclusive) and beyond 1000 km (exclusive) of the origin."""
    from groundfield.solver._frame import coordinate_span

    th = math.radians(37.0)
    world = _two_rods(
        (near * math.cos(th), near * math.sin(th)),
        (far * math.cos(th), far * math.sin(th)),
    )
    span = coordinate_span(world)
    assert span.mixed is flagged
    category = [d.severity for d in diagnose(world).by_category("coordinates")]
    # Not mixed, the 990 km extent is still reported as information.
    assert category == (["warning"] if flagged else ["info"])


# ---------------------------------------------------------------------
# The warning at solve time
# ---------------------------------------------------------------------


# audit W2B-04; user decision D5
@pytest.mark.parametrize(
    ("backend", "kw"),
    [
        ("image", {}),
        ("image", {"kernel": "point"}),
        ("mom", {}),
    ],
    ids=["image-line", "image-point", "mom"],
)
def test_the_solve_of_a_mixed_world_warns_once(backend, kw):
    # mom_sommerfeld runs the same point-matching runner as mom; its
    # quadrature of pairs 33 000 km apart takes about a minute, so it is
    # checked with the error filter in test_the_warning_comes_before_any_assembly.
    world = _mixed_world()
    engine = gf.create_engine(backend=backend, segment_length=0.5, **kw)
    result, mixed = _solve_recording(world, engine)
    assert len(mixed) == 1
    message = str(mixed[0].message)
    assert "'aux'" in message and "'h'" in message and "33 118 km" in message
    # The warning points at the caller, not into the package.
    assert Path(mixed[0].filename).resolve() == Path(__file__).resolve()
    # It only warns: the result is the world as given (Z_in is finite).
    assert np.isfinite(result.input_impedance()[0])


# audit W2B-04; user decision D5
def test_solve_mutual_warns_for_a_mixed_world():
    from groundfield.solver.mutual import solve_mutual_field, solve_mutual_matrix

    world = gf.create_world(soil=HOM)
    gf.create_electrode(
        world, "rod", name="f0", position=(UTM[0], UTM[1], 0.5), length=2.0
    )
    gf.create_electrode(world, "rod", name="f1", position=(20.0, 0.0, 0.5), length=2.0)
    engine = gf.create_engine(segment_length=0.5)
    for call in (
        lambda: solve_mutual_matrix(world, engine, ["f0", "f1"]),
        lambda: solve_mutual_field(world, engine, ["f0", "f1"], [[25.0, 0.0, 0.0]]),
    ):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            call()
        assert len(_mixed_records(rec)) == 1


# audit W2B-04; user decision D5
def test_the_warning_comes_before_any_assembly(monkeypatch):
    """The 33 118 km PEN with the default Sommerfeld earth return never finished.

    The warning is raised before the discretisation, so ``-W error`` (or a
    ``simplefilter("error")``) stops such a solve at once.
    """
    from groundfield.solver import _pipeline

    def not_reached(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("discretised before the coordinate check")

    monkeypatch.setattr(_pipeline, "discretise", not_reached)
    for backend, soil in (("image", HOM), ("mom", HOM), ("mom_sommerfeld", TWO)):
        world = _pen_to_local_cabinet()  # default earth return: sommerfeld
        world.soil = soil
        engine = gf.create_engine(backend=backend, segment_length=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MixedCoordinatesWarning)
            with pytest.raises(MixedCoordinatesWarning, match="'pen'"):
                engine.solve(world)


def test_the_warning_class_is_public_and_documented():
    import groundfield.diagnostics as diagnostics

    assert "MixedCoordinatesWarning" in diagnostics.__all__
    assert issubclass(MixedCoordinatesWarning, UserWarning)
    root = Path(gf.__file__).resolve().parents[2]
    page = (root / "docs/reference/warnings.md").read_text(encoding="utf-8")
    assert "### `MixedCoordinatesWarning`" in page
    assert "groundfield.diagnostics.MixedCoordinatesWarning" in page


# ---------------------------------------------------------------------
# Pure worlds and every generated network: nothing
# ---------------------------------------------------------------------


def _utm_ap1_like_world():
    """An AP1-like site from the layout builder, at UTM coordinates.

    A substation, six houses with foundations on both sides of a PEN cable
    of 180 m, every footprint and the substation given at UTM coordinates
    (an AP2 world built from projected survey data).
    """
    from groundfield.generators import LvNetworkLayout
    from groundfield.geo.footprint import BuildingFootprint

    ox, oy = UTM
    footprints = []
    for i in range(6):
        cx, cy = ox + 20.0 + 25.0 * i, oy + (18.0 if i % 2 == 0 else -18.0)
        footprints.append(
            BuildingFootprint(
                polygon_xy_m=[
                    (cx - 5.0, cy - 5.0),
                    (cx + 5.0, cy - 5.0),
                    (cx + 5.0, cy + 5.0),
                    (cx - 5.0, cy + 5.0),
                ],
                building_use="residential",
                osm_id=i,
            )
        )
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(ox, oy))
    layout.add_pen_cable(start="substation", end_xy=(ox + 180.0, oy))
    layout.connect_buildings()
    return layout.to_world(seed=0)


def _generated_networks():
    from groundfield.generators import (
        MeasurementSetupConfig,
        RadialTrunkTopology,
        TnNetworkConfig,
        TnNetworkGenerator,
    )

    configs = {
        "star_default": TnNetworkConfig(),
        "star_3": TnNetworkConfig(building_counts={"residential": 3}),
        "radial_10": TnNetworkConfig(
            building_counts={"residential": 10}, pen_topology=RadialTrunkTopology()
        ),
        "measurement": TnNetworkConfig(
            building_counts={"residential": 5}, measurement=MeasurementSetupConfig()
        ),
        "voltage_source": TnNetworkConfig(
            building_counts={"residential": 3}, source_kind="voltage"
        ),
    }
    return {
        name: TnNetworkGenerator(cfg, seed=0).build() for name, cfg in configs.items()
    }


def test_a_pure_utm_ap1_like_world_is_neither_reported_nor_warned_about():
    from groundfield.solver._frame import local_origin

    world = _utm_ap1_like_world()
    assert local_origin(world) is not None  # solved in the local frame
    engine = gf.create_engine(segment_length=1.0)
    assert not diagnose(world, engine).by_category("coordinates")
    _, mixed = _solve_recording(world, engine)
    assert mixed == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: _two_rods((0.0, 0.0), (40.0, 0.0)),
        lambda: _two_rods(UTM, (UTM[0] + 40.0, UTM[1])),
        # UTM without the zone prefix (easting 604 km, northing 5812 km).
        lambda: _two_rods((604_321.0, 5_812_345.0), (604_361.0, 5_812_345.0)),
        # A local world 99 km across, reaching 49.5 km from the origin.
        lambda: _two_rods((-49_500.0, 0.0), (49_500.0, 0.0)),
    ],
    ids=["local", "utm", "utm_without_zone_prefix", "local_99km"],
)
def test_pure_worlds_are_neither_reported_nor_warned_about(build):
    world = build()
    engine = gf.create_engine(segment_length=0.5)
    assert not diagnose(world, engine).by_category("coordinates")
    _, mixed = _solve_recording(world, engine)
    assert mixed == []


def test_every_generated_network_is_neither_reported_nor_warned_about():
    from groundfield.solver._frame import LocalFrame

    for name, world in _generated_networks().items():
        assert not diagnose(world).by_category("coordinates"), name
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            LocalFrame.of(world)
        assert _mixed_records(rec) == [], name
    # One generated network end to end through the solve.
    world = _generated_networks()["star_3"]
    _, mixed = _solve_recording(world, gf.create_engine(segment_length=1.0))
    assert mixed == []


def test_a_mixed_world_still_warns_after_a_json_round_trip():
    world = gf.World.model_validate_json(_mixed_world().model_dump_json())
    assert diagnose(world).by_category("coordinates")[0].severity == "warning"
    _, mixed = _solve_recording(world, gf.create_engine(segment_length=0.5))
    assert len(mixed) == 1


# ---------------------------------------------------------------------
# Generated networks with the substation at projected coordinates
# ---------------------------------------------------------------------


def _translated(world, shift_xy):
    """``world`` with every point moved horizontally by ``shift_xy`` (m)."""
    from groundfield.solver._frame import _moved

    shift = np.array([shift_xy[0], shift_xy[1], 0.0])
    return world.model_copy(
        update={
            field: [_moved(m, shift) for m in getattr(world, field)]
            for field in ("electrodes", "conductors", "junctions")
        }
    )


# audit W2R2-07
def test_w2r2_07_a_generated_network_with_its_substation_at_utm_is_mixed():
    """W2R2-07: the generators build in local coordinates.

    ``SubstationConfig.position`` moves the substation only; the houses,
    cabinets and the measurement stay around (0, 0), so a substation at UTM
    coordinates builds a mixed world (33 000 km PEN cables). Since D5 it is
    reported and warned about (a positive case of the check); the radial
    topology fails to build. Translating the whole built world instead gives
    a pure UTM world.
    """
    from groundfield.generators import (
        RadialTrunkTopology,
        TnNetworkConfig,
        TnNetworkGenerator,
    )
    from groundfield.generators.tn_network import SubstationConfig
    from groundfield.solver._frame import LocalFrame

    utm = (32_604_321.0, 5_812_345.0)
    counts = {"residential": 10}
    mixed = TnNetworkGenerator(
        TnNetworkConfig(
            substation=SubstationConfig(position=utm), building_counts=counts
        ),
        seed=0,
    ).build()
    (finding,) = diagnose(mixed).by_category("coordinates")
    assert finding.severity == "warning"
    # The substation (its ring, rods and bonds) is far, the houses are near.
    assert finding.data["far"] and all(
        name.startswith("trafo_") for name in finding.data["far"]
    )
    assert any(name.startswith("residential") for name in finding.data["near"])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        LocalFrame.of(mixed)
    assert len(_mixed_records(rec)) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="dropped every building"):
            TnNetworkGenerator(
                TnNetworkConfig(
                    substation=SubstationConfig(position=utm),
                    building_counts=counts,
                    pen_topology=RadialTrunkTopology(),
                ),
                seed=0,
            ).build()

    local = TnNetworkGenerator(TnNetworkConfig(building_counts=counts), seed=0).build()
    placed = _translated(local, utm)
    assert not diagnose(placed).by_category("coordinates")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        LocalFrame.of(placed)
    assert _mixed_records(rec) == []
