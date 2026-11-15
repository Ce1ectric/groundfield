"""Overlapping inductive conductors in ``diagnose()`` (roadmap item).

Since 2026.11 the inductance matrix evaluates conductors that lie beside each
other closer than touching (two cables drawn on one axis, a service drop drawn
on its PEN main) as touching wires; ``diagnose()`` reports such pairs as
``overlap`` warnings when the stretch exceeds 10 touching distances and 5 % of
the shorter conductor. Branches, crossings, T-contacts and a core inside its
cable shield are not overlaps. In the generated radial networks every drop of
a row of houses leaves the same tap on one axis: those overlaps are real
(spread 5 cm apart, Z_in of 5 houses moves by 3.5 % at 1 kHz, X by -15 %).
"""

from __future__ import annotations

import math
import warnings

import pytest

import groundfield as gf
from groundfield.diagnostics import (
    OVERLAP_FRACTION,
    OVERLAP_REL_LENGTH,
    DiagnosticsWarning,
)
from groundfield.generators import (
    HomogeneousSoilSpec,
    PenConfig,
    RadialTrunkTopology,
    StarCabinetTopology,
    TnNetworkConfig,
    TnNetworkGenerator,
)


def _cable(w, name, a, b, *, r=0.006, ctype="pen", model="neumann") -> None:
    gf.create_conductor(
        w,
        name=name,
        start=a,
        end=b,
        conductor_type=ctype,
        wire_radius=r,
        resistivity=2.82e-8,
        cross_section=95e-6,
        discretize_segment_length=10.0,
        coupling_to_soil="isolated",
        inductance_model=model,
    )


def _two_rods(length: float = 50.0) -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="a", position=(0.0, 0.0, 0.5), length=1.5)
    gf.create_electrode(w, "rod", name="b", position=(length, 0.0, 0.5), length=1.5)
    return w


def _overlaps(world: gf.World):
    return [d for d in gf.diagnose(world).findings if d.category == "overlap"]


def test_two_cables_on_one_axis_are_reported() -> None:
    w = _two_rods()
    _cable(w, "cable_1", "a", "b")
    _cable(w, "cable_2", "a", "b", r=0.004)
    (d,) = _overlaps(w)
    assert d.severity == "warning"
    assert d.subjects == ("cable_1", "cable_2")
    assert d.data["overlap_m"] == pytest.approx(50.0, rel=1e-9)
    assert d.data["distance_m"] == pytest.approx(0.0, abs=1e-12)
    assert d.data["touching_m"] == pytest.approx(0.010)
    text = str(gf.diagnose(w))
    assert "overlap: 1 pair of inductive conductors lies beside each other" in text
    with pytest.warns(DiagnosticsWarning, match="overlap"):
        gf.diagnose(w).warn()


def test_a_drop_drawn_back_along_its_main_is_reported() -> None:
    """The geometry of the golden case of the drops along the PEN main: the
    drop leaves the cabinet on the main's axis (they share the tap)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for name, x in (("s", 0.0), ("kvs", -200.0), ("h0", -20.0)):
        gf.create_electrode(w, "rod", name=name, position=(x, 0.0, 0.0), length=2.0)
    _cable(w, "pen_main", "s", "kvs")
    _cable(w, "drop_0", "kvs", "h0")
    (d,) = _overlaps(w)
    assert d.subjects == ("drop_0", "pen_main")
    # the stretch of the main within touching of the drop: 180 m plus 12 mm
    assert d.data["overlap_m"] == pytest.approx(180.012, rel=1e-6)


def test_cables_at_their_real_spacing_and_a_concentric_cable_are_not() -> None:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    for name, (x, y) in {
        "a1": (0, 0),
        "b1": (50, 0),
        "a2": (0, 0.02),
        "b2": (50, 0.02),
    }.items():
        gf.create_junction(w, name=name, position=(float(x), float(y), 0.7))
    gf.create_electrode(w, "rod", name="g", position=(0.0, 1.0, 0.5), length=1.5)
    _cable(w, "cable_1", "a1", "b1")
    _cable(w, "cable_2", "a2", "b2", r=0.004)
    assert _overlaps(w) == []
    w = _two_rods()
    _cable(w, "shield", "a", "b", r=0.015, ctype="cable_shield")
    _cable(w, "core", "a", "b", r=0.005)
    assert _overlaps(w) == []


def test_conductors_without_an_inductance_model_are_not_checked() -> None:
    w = _two_rods()
    _cable(w, "cable_1", "a", "b", model=None)
    _cable(w, "cable_2", "a", "b", model=None)
    assert _overlaps(w) == []


@pytest.mark.parametrize(
    ("angle_deg", "flagged"), [(10.0, False), (3.0, False), (0.2, True)]
)
def test_a_branch_is_an_overlap_only_when_it_runs_along(angle_deg, flagged) -> None:
    """A 30 m drop leaving its main at phi stays within touching (12 mm) for
    about 12 mm / phi: 0.07 m at 10 deg, 0.23 m at 3 deg (branches), 3.4 m at
    0.2 deg, more than 5 % of the drop (an overlap)."""
    w = _two_rods(100.0)
    t = math.radians(angle_deg)
    gf.create_electrode(
        w,
        "rod",
        name="h",
        position=(30 * math.cos(t), 30 * math.sin(t), 0.5),
        length=1.5,
    )
    _cable(w, "main", "a", "b")
    _cable(w, "drop", "a", "h")
    found = _overlaps(w)
    assert bool(found) is flagged
    if flagged:
        stretch = found[0].data["overlap_m"]
        assert stretch > max(OVERLAP_REL_LENGTH * 0.012, OVERLAP_FRACTION * 30.0)
        assert stretch == pytest.approx(0.012 / math.sin(t), rel=0.05)


# audit W3B-03
def test_w3b_03_cables_drawn_at_the_touching_distance_are_not_an_overlap() -> None:
    """Base 098bbbc: two 40 m cables (5 mm) drawn 1e-14 m closer than
    touching (round-off of a drawing at the touching distance) were reported
    as overlapping over 40 m, although the inductance matrix evaluates them
    the same either way; 6 of the 31 overlaps of the 80-house radial network
    stayed when its drops were drawn side by side at the touching distance.
    The check has a relative tolerance of 1e-9 at the touching distance."""
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    y0 = 1.234567
    for k, y in enumerate((y0, y0 + 0.01 * (1.0 - 1e-12))):
        gf.create_junction(w, name=f"a{k}", position=(0.0, y, 0.7))
        gf.create_junction(w, name=f"b{k}", position=(40.0, y, 0.7))
        w.add_conductor(
            gf.Conductor(
                name=f"c{k}",
                start=(0.0, y, 0.7),
                end=(40.0, y, 0.7),
                start_electrode=f"a{k}",
                end_electrode=f"b{k}",
                wire_radius=0.005,
                coupling_to_soil="isolated",
                inductance_model="neumann",
            )
        )
    assert not _overlaps(w)


# audit W3B-03
@pytest.mark.parametrize(
    ("factor", "flagged"), [(1.0 - 1e-6, True), (1.0 - 1e-10, False)]
)
def test_w3b_03_the_tolerance_is_relative_and_tight(factor, flagged) -> None:
    """1e-6 inside the touching distance is still an overlap; 1e-10 is
    round-off (the tolerance is 1e-9 of the touching distance)."""
    w = _two_rods()
    for k, y in enumerate((0.0, 0.012 * factor)):
        gf.create_junction(w, name=f"a{k}", position=(0.0, y, 0.7))
        gf.create_junction(w, name=f"b{k}", position=(50.0, y, 0.7))
        _cable(w, f"cable_{k}", f"a{k}", f"b{k}")
    assert bool(_overlaps(w)) is flagged


def _generated(n: int, topology) -> gf.World:
    cfg = TnNetworkConfig(
        soil=HomogeneousSoilSpec(rho=100.0),
        building_counts={"residential": n},
        pen_topology=topology,
        pen=PenConfig(inductance_model="neumann", segment_length_m=5.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TnNetworkGenerator(cfg, seed=0).build()


@pytest.mark.parametrize(
    ("n", "topology", "expected"),
    [
        (30, StarCabinetTopology(), 0),
        (200, StarCabinetTopology(), 0),
        (5, RadialTrunkTopology(), 10),
        (30, RadialTrunkTopology(), 2),
    ],
    ids=["star30", "star200", "radial5", "radial30"],
)
def test_generated_networks_report_only_their_drops_on_one_axis(
    n, topology, expected
) -> None:
    """Measured on the generated networks (stream w3perf, item 4): the star
    networks of 30 and 200 houses have none; in the radial ones the drops of
    houses behind each other leave one tap on one axis (5 houses: all five
    drops from the substation ring along -x, 10 pairs)."""
    found = _overlaps(_generated(n, topology))
    assert len(found) == expected
    for d in found:
        assert all(name.startswith("pen_service_") for name in d.subjects)
        assert d.data["distance_m"] < 1e-9
