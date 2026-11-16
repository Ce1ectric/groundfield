r"""Uniform-current owners of generated foundation electrodes.

The image family solves for one uniform leakage current per straight wire
(an "owner"; an outline is one owner per straight side, user decisions Q30,
Q44). Up to 2026.11 the generator built an axis-aligned bare mesh
foundation as one ``grid_mesh`` electrode (one owner over perimeter and
braces) but a rotated one, and since 10.12-10.14 every encased one, as one
electrode per wire. So a shell that is electrically absent
(:math:`\rho_c = \rho_\text{soil}`, or 0.1 mm thick) moved Z of the default
foundation by -1.2 % and the largest touch voltage by -4.8 % (audit A2G-04),
and the same bare foundation differed by 1.2 % between 0 and 30 degrees
(10.39). The single owner is the outlier: against the equipotential
reference (one owner per segment; equal to ``mom`` point matching) it is
+1.5 / +2.4 % high for 2 x 2 / 3 x 3 cells, one owner per wire +0.3 /
+0.4 % (evidence ``mesh_owner_granularity.out``).

Rule since 2026.11 (the owners depend only on the structure: never on the
orientation, on whether a shell is configured or on ``concrete_model``):

* a mesh foundation (cross-braces) has one owner per wire line;
* a rectangle without cross-braces (``style="ring"``, or ``"mesh"`` with
  ``n_x = n_y = 1``) has one owner per side (user decision Q38). Up to then
  the default ``"lumped"`` made an axis-aligned one a single ``grid_mesh``
  owner and only ``"distributed"`` (or a rotation) split it; per side its
  input impedance is lower, by less than 1 % up to aspect ratio 4 and 2 %
  up to 50 over the AP1 soils for a short side of at least 1 m, bare or in
  concrete up to 0.2 m thick (audit D2B-02, R2B-01), and equal for a
  square;
* a ``polygon_xy_m`` outline is one closed polyline, one owner per straight
  side (user decisions Q30, Q44).

``concrete_model`` therefore has no effect. It is deprecated (user
decisions Q21 and Q38): ``"distributed"`` warns at the user's line, the
default and the files of 0.15.0 load silently, and the generators do not
repeat the warning.
"""

from __future__ import annotations

import json
import math
import pathlib
import pickle
import re
import sys
import warnings

import numpy as np
import pydantic
import pytest

import groundfield as gf
from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec
from groundfield.generators.electrode_specs import _CONCRETE_MODEL_DEPRECATION

RHO = 100.0


def _site(spec: FoundationElectrodeSpec, soil: object | None = None) -> gf.World:
    world = gf.World(soil=soil if soil is not None else gf.HomogeneousSoil(rho=RHO))
    anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
        world, (0.0, 0.0), "h", np.random.default_rng(0)
    )
    world.add_source(gf.CurrentSource(name="s", attached_to=anchor, magnitude=1.0))
    return world


def _z(world: gf.World, kernel: str = "line", backend: str = "image") -> float:
    kwargs = {"kernel": kernel} if backend == "image" else {}
    engine = gf.create_engine(
        backend=backend, segment_length=1.0, frequencies=[50.0], **kwargs
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(engine.solve(world).input_impedance()[0]).real


def _mesh(**kw: object) -> FoundationElectrodeSpec:
    params: dict[str, object] = dict(
        style="mesh", size_m=10.0, depth_m=0.7, wire_radius_m=0.005
    )
    params.update(kw)
    return FoundationElectrodeSpec(**params)


def _ring(**kw: object) -> FoundationElectrodeSpec:
    params: dict[str, object] = dict(
        style="ring", size_xy_m=(20.0, 2.0), depth_m=0.7, wire_radius_m=0.005
    )
    params.update(kw)
    return FoundationElectrodeSpec(**params)


def _owners(world: gf.World) -> list:
    return [e for e in world.electrodes if e.name.startswith("h_foundation")]


# ---------------------------------------------------------------------
# A shell that is electrically absent changes nothing
# ---------------------------------------------------------------------


# audit A2G-04
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("kernel", ["line", "point"])
@pytest.mark.parametrize("angle", [None, 30.0])
@pytest.mark.parametrize("thickness_m", [0.05, 1e-4], ids=["t=5cm", "t=0.1mm"])
def test_vanishing_shell_on_a_mesh_foundation_is_continuous(
    model: str, kernel: str, angle: float | None, thickness_m: float
) -> None:
    """rho_c = rho_soil is transparent by construction (bare wire of radius
    r_b plus C / L = bare wire of radius r_a); 2026.11 before the fix: -1.27 %
    (line) / -1.18 % (point) for the axis-aligned mesh at any thickness."""
    bare = _z(_site(_mesh(orientation_deg=angle, concrete_model=model)), kernel)
    shell = _z(
        _site(
            _mesh(
                orientation_deg=angle,
                concrete_model=model,
                concrete_rho_ohm_m=RHO,
                concrete_thickness_m=thickness_m,
            )
        ),
        kernel,
    )
    # The line kernel's self term is not exactly logarithmic in the
    # radius: -0.05 % at r_b = 5.5 cm, nothing at 0.1 mm.
    tolerance = 1e-9 if kernel == "point" else (1e-3 if thickness_m > 1e-3 else 1e-5)
    assert shell == pytest.approx(bare, rel=tolerance)


# audit A2G-04, user decision Q38
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("thickness_m", [0.05, 1e-4], ids=["t=5cm", "t=0.1mm"])
def test_vanishing_shell_on_a_ring_is_continuous(
    model: str, thickness_m: float
) -> None:
    """A bare ``"distributed"`` ring was one owner, an encased one four:
    -0.24 % for a 20 x 2 m ring. Since user decision Q38 every ring is four
    owners for both values (the test ran for ``"distributed"`` only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        bare = _z(_site(_ring(concrete_model=model)), "point")
        shell = _z(
            _site(
                _ring(
                    concrete_model=model,
                    concrete_rho_ohm_m=RHO,
                    concrete_thickness_m=thickness_m,
                )
            ),
            "point",
        )
    assert shell == pytest.approx(bare, rel=1e-9)


# ---------------------------------------------------------------------
# The orientation and the description change nothing
# ---------------------------------------------------------------------


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("size_xy_m", [(10.0, 10.0), (16.0, 8.0)])
@pytest.mark.parametrize("rho_c", [None, 1000.0])
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_mesh_foundation_does_not_depend_on_its_orientation(
    size_xy_m: tuple[float, float], rho_c: float | None, model: str
) -> None:
    """Bare 10 x 10 m mesh before the fix: 4.79618 ohm at 0 deg, 4.73771 ohm
    at 30 deg (-1.2 %)."""
    values = [
        _z(
            _site(
                _mesh(
                    size_xy_m=size_xy_m,
                    orientation_deg=angle,
                    concrete_rho_ohm_m=rho_c,
                    concrete_model=model,
                )
            )
        )
        for angle in (None, 30.0, 90.0, -45.0)
    ]
    # Rotation invariance of the image sums holds to about 1e-7 (the
    # near-pair line integrals of axis-parallel and oblique segments are
    # evaluated differently); the artefact was 1e-2.
    assert values == pytest.approx([values[0]] * 4, rel=1e-6)


# audit A2G-04, user decision Q38
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("rho_c", [None, 1000.0])
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_ring_does_not_depend_on_its_orientation(
    rho_c: float | None, model: str
) -> None:
    """A bare distributed ring was one owner at 0 deg and four at 30 deg.
    Up to user decision Q38 the default ``"lumped"`` kept the axis-aligned
    ring (and the one at 90 deg) one grid-mesh owner, 0.24 % above the
    rotated one for this 20 x 2 m ring (the documented exception, pinned by
    ``test_lumped_ring_keeps_its_axis_aligned_single_owner``, removed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        values = [
            _z(
                _site(
                    _ring(
                        orientation_deg=angle,
                        concrete_model=model,
                        concrete_rho_ohm_m=rho_c,
                    )
                )
            )
            for angle in (None, 30.0, 90.0)
        ]
    assert values == pytest.approx([values[0]] * 3, rel=1e-6)


# ---------------------------------------------------------------------
# The owners themselves
# ---------------------------------------------------------------------


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("angle", [None, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_mesh_foundation_is_one_owner_per_wire(
    angle: float | None, rho_c: float | None, model: str
) -> None:
    world = _site(
        _mesh(orientation_deg=angle, concrete_rho_ohm_m=rho_c, concrete_model=model)
    )
    owners = _owners(world)
    assert [e.name for e in owners] == ["h_foundation_0"] + [
        f"h_foundation_0_w{j}" for j in range(1, 6)
    ]
    assert {e.kind for e in owners} == {"polyline"}
    assert sum(e.total_length() for e in owners) == pytest.approx(60.0)
    bonds = [c for c in world.conductors if c.name.startswith("h_foundation_0_bond")]
    assert len(bonds) == 5


# audit A2G-04, user decision Q38
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("angle", [None, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_ring_owners(angle: float | None, rho_c: float | None, model: str) -> None:
    """Four strips for both values at every angle; up to user decision Q38
    the axis-aligned ``"lumped"`` ring was one ``grid_mesh``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spec = _ring(
            orientation_deg=angle, concrete_rho_ohm_m=rho_c, concrete_model=model
        )
    owners = _owners(_site(spec))
    assert [e.kind for e in owners] == ["strip"] * 4
    assert sum(e.total_length() for e in owners) == pytest.approx(44.0)


_SIDES = ["h_foundation_0"] + [f"h_foundation_0_w{j}" for j in (1, 2, 3)]


# user decision Q38
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("style", ["ring", "mesh"])
@pytest.mark.parametrize("angle", [None, 90.0, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
def test_rectangle_without_braces_is_one_owner_per_side(
    style: str, angle: float | None, rho_c: float | None
) -> None:
    """Every rectangle without cross-braces is one strip per side, bonded to
    the anchor as a rotated one, with the default value too. Up to user
    decision Q38 the default made an axis-aligned one (also at 90 deg, which
    was realised as the axis-aligned rectangle with swapped sides up to
    audit D3B-04) one ``grid_mesh`` connected at its centre."""
    spec = FoundationElectrodeSpec(
        style=style,
        size_xy_m=(20.0, 2.0),
        n_x=1,
        n_y=1,
        depth_m=0.7,
        wire_radius_m=0.005,
        orientation_deg=angle,
        concrete_rho_ohm_m=rho_c,
    )
    world = _site(spec)
    owners = _owners(world)
    assert [e.name for e in owners] == _SIDES
    assert [e.kind for e in owners] == ["strip"] * 4
    assert sum(e.total_length() for e in owners) == pytest.approx(44.0)
    bonds = [
        (c.name, c.start_electrode, c.end_electrode)
        for c in world.conductors
        if c.name.startswith("h_foundation_0_bond")
    ]
    assert bonds == [
        (f"h_foundation_0_bond_{j}", "h_foundation_0", f"h_foundation_0_w{j}")
        for j in (1, 2, 3)
    ]
    radius = 0.005 if rho_c is None else 0.055
    assert [e.wire_radius for e in owners] == pytest.approx([radius] * 4)
    coefficients = {e.concrete_shell_coefficient_ohm_m for e in owners}
    assert len(coefficients) == 1 and (coefficients.pop() > 0.0) == (rho_c is not None)
    # The anchor is the local -y side; it starts at the local (-dx/2, -dy/2)
    # corner, the connection point of conductors attached by name (it was the
    # centre of the grid mesh). At 90 deg the rectangle is rotated exactly
    # (audit D3B-04, user decision Q41): the corner is (1, -10) m, no longer
    # the (-1, -10) m corner of the axis-aligned rectangle with swapped sides.
    corner = {None: (-10.0, -1.0), 90.0: (1.0, -10.0)}.get(angle)
    if corner is not None:
        assert owners[0].connection_point == pytest.approx((*corner, 0.7))


def _solved(spec: FoundationElectrodeSpec) -> dict[str, object]:
    """Every result of one generated foundation that a user reads."""
    world = _site(spec, gf.TwoLayerSoil(rho_1=200.0, rho_2=50.0, h_1=4.0))
    engine = gf.create_engine(frequencies=[50.0, 1000.0])
    points = np.array(
        [[x, y, 0.0] for x in (-14.0, -3.0, 0.0, 7.0) for y in (-4.0, 0.0, 2.5)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = engine.solve(world)
        _, touch, feet = gf.touch_voltage_envelope(
            result,
            world,
            electrode="h_foundation_0",
            interior="meshes",
            return_points=True,
        )
        return {
            "fingerprint": gf.world_fingerprint(world),
            "input impedance": np.asarray(result.input_impedance()),
            "cluster impedance": np.asarray(result.cluster_impedance("h_foundation_0")),
            "potentials": np.array([result.electrode_potentials[n] for n in _SIDES]),
            "currents": np.array([result.electrode_currents[n] for n in _SIDES]),
            "field": np.array([result.potential(points, k) for k in (0, 1)]),
            "touch voltages": np.asarray(touch),
            "feet": np.asarray(feet),
        }


# user decision Q38
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("angle", [None, 90.0, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
def test_both_concrete_model_values_give_the_same_results(
    angle: float | None, rho_c: float | None
) -> None:
    """``concrete_model`` has no effect: the same world, bit for bit the same
    impedances, potentials, currents, field and touch voltages (the
    envelope with its interior sample, the centre of the rectangle). Up to
    user decision Q38 the two values differed for an axis-aligned rectangle
    (0.24 % for this 20 x 2 m one in homogeneous soil)."""
    results = {}
    for model in ("lumped", "distributed"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = _ring(
                orientation_deg=angle, concrete_rho_ohm_m=rho_c, concrete_model=model
            )
        results[model] = _solved(spec)
    lumped, distributed = results["lumped"], results["distributed"]
    assert lumped["fingerprint"] == distributed["fingerprint"]
    for key, value in lumped.items():
        if key != "fingerprint":
            assert np.array_equal(value, distributed[key]), key
    # 24 directions on the offset curve plus the one cell centre
    # 24 equally spaced feet, the one cell centre, 5 feet on each corner arc
    assert len(lumped["touch voltages"]) == 24 + 1 + 4 * 5


# user decision Q38
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_json_round_trip_of_either_value_builds_the_same_foundation(
    model: str,
) -> None:
    """A spec written with either value loads back with it (``"distributed"``
    with the warning, ``"lumped"`` silently) and builds the world of a spec
    without the argument."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        spec = _ring(concrete_rho_ohm_m=150.0, concrete_model=model)
    text = spec.model_dump_json()
    assert json.loads(text)["concrete_model"] == model
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = FoundationElectrodeSpec.model_validate_json(text)
    assert len(_concrete_model_warnings(caught)) == (model == "distributed")
    assert restored == spec and restored.concrete_model == model
    reference = _site(_ring(concrete_rho_ohm_m=150.0))
    assert gf.world_fingerprint(_site(restored)) == gf.world_fingerprint(reference)


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("rho_c", [None, 150.0])
def test_polygon_outline_is_one_closed_wire(model: str, rho_c: float | None) -> None:
    outline = (
        (-6.0, -4.5),
        (6.0, -4.5),
        (6.0, 4.5),
        (-1.0, 4.5),
        (-1.0, 1.5),
        (-6.0, 1.5),
    )
    world = _site(
        _ring(polygon_xy_m=outline, concrete_model=model, concrete_rho_ohm_m=rho_c)
    )
    (owner,) = _owners(world)
    assert owner.kind == "polyline" and owner.closed
    assert owner.total_length() == pytest.approx(42.0)


# ---------------------------------------------------------------------
# Why one owner per wire: the equipotential reference
# ---------------------------------------------------------------------


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize(
    ("size_m", "cells", "limit"),
    [(10.0, 2, 0.004), (20.0, 3, 0.006)],
)
def test_mesh_foundation_is_close_to_the_equipotential_reference(
    size_m: float, cells: int, limit: float
) -> None:
    """``mom`` (one current per segment, point matching) is the
    equipotential reference of the same segments. One owner over perimeter
    and braces was +1.47 % (2 x 2) / +2.44 % (3 x 3) above it, one owner
    per wire is +0.28 / +0.45 %."""
    world = _site(_mesh(size_m=size_m, n_x=cells, n_y=cells))
    z_image = _z(world, kernel="point")
    z_mom = _z(world, backend="mom")
    assert 0.0 < z_image / z_mom - 1.0 < limit


# audit A3SG-06
@pytest.mark.finding("10.39")
@pytest.mark.parametrize(("size_m", "cells"), [(10.0, 2), (20.0, 3)])
def test_documented_margin_of_mesh_foundations_against_mom(
    size_m: float, cells: int
) -> None:
    """The docs (generators page, known limitations) state the margin
    against ``mom`` for the default kernel: +1.1 %; with the point kernel
    (the kernel of ``mom``) +0.3 ... +0.5 %. They said +0.3 ... +0.6 %
    "against mom" for the default kernel, which is the line kernel against
    a line-kernel per-segment reference. Measured (100 ohm m, 1 m
    segments): line +1.13 / +1.13 %, point +0.28 / +0.45 %; mom at 0.5 m
    segments differs from mom at 1 m by +0.02 / +0.04 %."""
    world = _site(_mesh(size_m=size_m, n_x=cells, n_y=cells))
    z_mom = _z(world, backend="mom")
    assert _z(world) / z_mom - 1.0 == pytest.approx(0.0113, abs=0.0005)
    assert 0.0025 < _z(world, kernel="point") / z_mom - 1.0 < 0.005


# ---------------------------------------------------------------------
# concrete_model is deprecated and has no effect (user decisions Q21, Q38)
# ---------------------------------------------------------------------

V0150_CONFIG = pathlib.Path(__file__).parent / "data" / "v0150_tn_network_config.json"
V0150_PICKLES = pathlib.Path(__file__).parent / "data" / "v0150_generator_pickles.pkl"


def _concrete_model_warnings(caught: list) -> list[str]:
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, DeprecationWarning)
        and "concrete_model" in str(w.message)
    ]


def _distributed_ring(**kw: object) -> FoundationElectrodeSpec:
    """A ring with the deprecated value, its warning consumed."""
    with pytest.warns(DeprecationWarning, match="concrete_model"):
        return _ring(concrete_model="distributed", **kw)


# user decisions Q21 and Q38
@pytest.mark.finding("10.12")
def test_distributed_concrete_model_warns_at_the_callers_line() -> None:
    """The non-default value keeps working and warns: that it has no effect
    since 2026.11, what every rectangle without cross-braces is now, the
    earliest removal, what to do. Up to user decision Q38 the message said
    that the value chose one owner per side for an axis-aligned rectangle
    and quoted how far the two values differed."""
    with pytest.warns(DeprecationWarning, match="concrete_model") as caught:
        spec = _ring(concrete_model="distributed")
    (warning,) = [w for w in caught if "concrete_model" in str(w.message)]
    assert warning.filename == __file__  # the user's line, not groundfield's
    message = " ".join(str(warning.message).split())
    for part in (
        "has no effect since groundfield 2026.11",
        # audit D2B-05: every foundation the owner rule covers
        "every rectangle without cross-braces (style='ring', or style='mesh' "
        "with n_x = n_y = 1) is one uniform-current owner per side",
        "at every orientation",
        "what 'distributed' did",
        "2027.5",
        "drop the argument",
    ):
        assert part in message, part
    assert "differ" not in message and " % " not in message
    assert spec.concrete_model == "distributed"
    world = _site(spec)
    assert [e.kind for e in _owners(world)] == ["strip"] * 4


def _quoted_bounds(text: str) -> dict[str, float]:
    """The bound a text quotes for the change: all shells, the default shell."""
    found = _SCOPE.search(" ".join(text.split()))
    assert found, text
    return {
        "4": float(found.group(1)),
        "50": float(found.group(3)),
        "4-default": float(found.group(5)),
        "50-default": float(found.group(6)),
    }


def _per_side_over_one_owner(
    size_xy_m: tuple[float, float],
    soil: object,
    rho_c: float | None,
    thickness_m: float,
    style: str = "ring",
) -> float:
    """Z(one owner per side) / Z(one owner) - 1 of a rectangle without
    cross-braces (default engine, 0.8 m, 5 mm wire): the generated
    foundation against one ``GridMeshElectrode`` with the same radius and
    shell coefficient, the world the generators built for the default value
    up to user decision Q38 (evidence d3gen ``q38_change_cases.py``: the same
    world fingerprint on the base). Since user decision Q30 the image family
    gives that grid one owner per wire line as well; its one-owner model is
    rebuilt with the solver's private comparison switch."""
    from groundfield.geometry.electrodes import concrete_shell_coefficient
    from groundfield.solver._discretize import _one_owner_per_wire_off

    spec = FoundationElectrodeSpec(
        style=style,
        size_xy_m=size_xy_m,
        n_x=1,
        n_y=1,
        depth_m=0.8,
        wire_radius_m=0.005,
        concrete_rho_ohm_m=rho_c,
        concrete_thickness_m=thickness_m,
    )
    per_side = _site(spec, soil)
    one_owner = gf.World(soil=soil)
    dx, dy = size_xy_m
    gf.create_electrode(
        one_owner,
        "grid_mesh",
        name="h_foundation_0",
        corner=(-dx / 2.0, -dy / 2.0, 0.8),
        size=(dx, dy),
        n_x=1,
        n_y=1,
        wire_radius=0.005 if rho_c is None else 0.005 + thickness_m,
        concrete_shell_coefficient_ohm_m=(
            0.0
            if rho_c is None
            else concrete_shell_coefficient(rho_c, 0.005, thickness_m)
        ),
    )
    one_owner.add_source(
        gf.CurrentSource(name="s", attached_to="h_foundation_0", magnitude=1.0)
    )
    engine = gf.create_engine(frequencies=[50.0])
    values = []
    for world in (per_side, one_owner):
        with warnings.catch_warnings(), _one_owner_per_wire_off():
            warnings.simplefilter("ignore")
            values.append(complex(engine.solve(world).input_impedance()[0]).real)
    return values[0] / values[1] - 1.0


# audit D2B-02, audit R2B-01, audit R2B-02, user decision Q38
@pytest.mark.finding("10.39")
@pytest.mark.parametrize(
    ("size_xy_m", "soil", "rho_c", "thickness_m", "bound", "measured"),
    [
        # the largest value found in each clause of the scope (evidence f3gen
        # r2b_scope_scan.py, r2b_scope_refine.py; signed: d3gen
        # q38_signed_analyse.out, q38_change_cases_*.out). Remeasured with
        # the line kernel of wave 3 (stream w3xl: blended near/far switch,
        # far images with the quadrupole term): moved by up to 1.7e-5; and
        # with the transverse distance of image pairs (stream fw3xl, audit
        # W3A-02; the 0.205 m shells): by up to 8.0e-8.
        ((4.0, 1.0), (1000.0, 30.0, 5.0), 30.0, 0.2, "4", -0.00687543),
        ((16.0, 1.0), (1000.0, 30.0, 10.0), 30.0, 0.2, "50", -0.01527894),
        ((80.0, 20.0), (30.0, 200.0, 5.0), 30.0, 0.05, "4-default", -0.00319689),
        ((150.0, 7.75), (30.0, 200.0, 5.0), None, 0.05, "50-default", -0.00906735),
        ((7.0, 1.75), (100.0, 100.0, None), None, 0.05, "4-default", -0.00088626),
        ((25.0, 1.0), (100.0, 100.0, None), None, 0.05, "50-default", -0.00329719),
        # ... the cases with which the auditor broke the earlier bounds
        ((10.0, 2.5), (100.0, 100.0, None), None, 0.05, "4-default", -0.00084106),
        ((40.0, 10.0), (30.0, 200.0, 5.0), None, 0.05, "4-default", -0.00252475),
        ((50.0, 1.0), (1000.0, 30.0, 30.0), 30.0, 0.2, "50", -0.01224330),
        # ... and a square, which does not change (symmetry)
        ((10.0, 10.0), (1000.0, 30.0, 5.0), 30.0, 0.2, "4", 0.0),
    ],
    ids=[
        "4x1-wet-t0.2",
        "16x1-wet-t0.2",
        "80x20-wet-t5cm",
        "150x7.75-bare",
        "7x1.75-hom",
        "25x1-hom",
        "10x2.5-hom",
        "40x10-bare",
        "50x1-wet-t0.2",
        "10x10-wet-t0.2",
    ],
)
def test_the_quoted_size_of_the_owner_change_holds(
    size_xy_m: tuple[float, float],
    soil: tuple[float, float, float | None],
    rho_c: float | None,
    thickness_m: float,
    bound: str,
    measured: float,
) -> None:
    """User decision Q38 turned the bound between the two ``concrete_model``
    values into the size of this release's change for axis-aligned
    rectangles without cross-braces built with the default value: one owner
    became one owner per side, and per side is never higher (the one-owner
    current distribution is one of those the four owners can take, and the
    bonded owners minimise the dissipation; equal for a square). The bound
    keeps its scope and margin (audit D2B-02, R2B-01/02: the earlier
    quotes broke at 0.38 / 1.24 %, then at 16 m x 1 m 1.553 % and 25 m x 1 m
    0.331 %, as lumped / distributed - 1); the quoted texts state it as a
    decrease, and a mesh without braces changes like the ring (D2B-05)."""
    r1, r2, h = soil
    model = (
        gf.HomogeneousSoil(rho=r1)
        if h is None
        else gf.TwoLayerSoil(rho_1=r1, rho_2=r2, h_1=h)
    )
    change = _per_side_over_one_owner(size_xy_m, model, rho_c, thickness_m)
    assert change == pytest.approx(measured, abs=1e-8 if measured else 1e-12)
    assert change <= 1e-12
    for text in _texts_with_the_scope().values():
        assert abs(change) < _quoted_bounds(text)[bound] / 100.0
    mesh = _per_side_over_one_owner(size_xy_m, model, rho_c, thickness_m, "mesh")
    assert mesh == change


_DOCS = pathlib.Path(__file__).parents[1] / "docs"
_SCOPE = re.compile(
    r"lower by less than ([0-9.]+) % up to aspect ratio 4 \(short side at least "
    r"([0-9.]+) m, sides up to 80 m\) and ([0-9.]+) % up to aspect ratio 50 "
    r"\(short side at least ([0-9.]+) m, sides up to 150 m\); bare or with the "
    r"default 5 cm shell by less than ([0-9.]+) % and ([0-9.]+) %"
)


def _texts_with_the_scope() -> dict[str, str]:
    """Every text that quotes the size of the change with its scope,
    whitespace folded. Up to user decision Q38 the warning and the
    deprecations page quoted it too, as the bound between the two values;
    they now only say that the field has no effect."""
    from groundfield.generators import materialise

    def read(*parts: str) -> str:
        return _DOCS.joinpath(*parts).read_text(encoding="utf-8")

    adr = read("adr", "0012-foundation-concrete-encasement.md")
    page = read("explanation", "concrete-shells.md")
    texts = {
        "field description": FoundationElectrodeSpec.model_fields[
            "concrete_model"
        ].description,
        "materialise module": materialise.__doc__,
        "ADR-0012": adr.split("### User decision Q38", 1)[1],
        "concrete-shell page": page.split(
            "## Owners of a rectangle without cross-braces", 1
        )[1].split("\n## ", 1)[0],
    }
    return {name: " ".join(text.split()) for name, text in texts.items()}


# audit R2B-01, audit R2B-02, user decision Q38
@pytest.mark.finding("10.39")
def test_every_text_quotes_the_size_of_the_change_with_the_short_side() -> None:
    """The aspect-ratio-4 clause named no shortest side, but the scan behind
    it started at 4 m x 1 m: narrower rectangles exceed the bound (2 m x
    0.5 m: 1.0157 % as lumped / distributed - 1, -1.0056 % as the change
    one owner -> per side, with 0.2 m of wet concrete over 1000 / 30 ohm m).
    Every text quotes one bound, with a short side of at least 1 m in both
    clauses, and the short summaries give the same numbers. Since user
    decision Q38 it is the size of the change one owner -> one owner per
    side, a decrease ("lower by"); up to then it was the difference between
    the two values of ``concrete_model`` (this test was
    ``test_every_text_quotes_the_bound_with_the_short_side_in_both_clauses``)."""
    quoted, problems = {}, []
    for name, text in _texts_with_the_scope().items():
        found = _SCOPE.search(text)
        if found is None:
            problems.append(f"{name}: no decrease with the short side in both clauses")
        else:
            quoted[name] = tuple(float(g) for g in found.groups())
        if "bare in homogeneous soil by less than" in text:
            problems.append(f"{name}: a bare-homogeneous bound without margin")
    summary = (
        "lower by less than 1 % up to aspect ratio 4 and 2 % up to 50 for a "
        "short side of at least 1 m"
    )
    generators_page = (_DOCS / "reference" / "api" / "generators.md").read_text(
        encoding="utf-8"
    )
    for name, text in {
        "class docstring": FoundationElectrodeSpec.__doc__,
        "generators page": generators_page,
    }.items():
        if summary not in " ".join(text.split()):
            problems.append(f"{name}: no short summary with the short side")
    assert not problems, problems
    assert set(quoted.values()) == {(1.0, 1.0, 2.0, 1.0, 0.4, 1.0)}, quoted
    # The short side matters: below 1 m the bound of aspect ratio 4 fails.
    narrow = _per_side_over_one_owner(
        (2.0, 0.5), gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 30.0, 0.2
    )
    assert narrow == pytest.approx(-0.01005558, abs=1e-8)  # fw3xl: -0.01005527
    assert abs(narrow) > quoted["field description"][0] / 100.0


# audit R2B-07, user decision Q38
@pytest.mark.finding("10.39")
def test_the_owner_section_states_its_model_uncertainty() -> None:
    """The largest values of the change sit at the 0.2 m shell, whose outer
    radius (0.205 m) makes segment length / wire radius 2.4 at the default
    0.5 m segments, below the thin-wire limit of 5 (the solve warns). There an
    independent solver that averages the potential over the wire surface
    gives -0.594 / -1.447 % instead of -0.686 / -1.529 % (0.598 / 1.468 %
    against 0.691 / 1.553 % as lumped / distributed - 1, as the texts said
    up to user decision Q38), and the comparison with one current per segment
    is not converged."""
    spec = FoundationElectrodeSpec(
        style="ring",
        size_xy_m=(16.0, 1.0),
        depth_m=0.8,
        wire_radius_m=0.005,
        concrete_rho_ohm_m=30.0,
        concrete_thickness_m=0.2,
    )
    world = _site(spec, gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=10.0))
    with pytest.warns(
        UserWarning, match=r"segment_length / wire_radius below 5.* 2\.4"
    ):
        gf.create_engine(frequencies=[50.0]).solve(world)
    texts = _texts_with_the_scope()
    page, adr = texts["concrete-shell page"], texts["ADR-0012"]
    assert "reproduces the largest cases to 0.01 points" not in page
    for text in (page, adr):
        assert "below the thin-wire limit of 5" in text
        assert "−0.594" in text and "−1.447" in text
        assert "−0.686" in text and "−1.529" in text
    assert "segment length over wire radius is 2.4" in page
    assert "is not converged" in adr
    assert "refining its segments to 0.083 m" in page


# user decision Q38 (replaces audit D2B-05's
# test_the_texts_name_every_foundation_whose_owners_concrete_model_changes:
# the value changes no owner any more)
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("angle", [None, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
def test_concrete_model_has_no_effect_on_any_foundation(
    angle: float | None, rho_c: float | None
) -> None:
    """Rings, meshes without and with braces and polygon outlines build the
    same world for both values. Up to user decision Q38 a ``style="mesh"``
    foundation with ``n_x = n_y = 1`` was one grid mesh with ``"lumped"`` and
    four strips with ``"distributed"``, like a ring (audit D2B-05); the
    warning and the field description now say that the field has no effect
    and name every foundation of the owner rule."""
    outline = ((-6.0, -4.5), (6.0, -4.5), (6.0, 4.5), (-6.0, 4.5))
    variants = {
        "ring": dict(style="ring"),
        "mesh 1 x 1": dict(style="mesh", n_x=1, n_y=1),
        "mesh 1 x 2": dict(style="mesh", n_x=1, n_y=2),
        "mesh 2 x 2": dict(style="mesh"),
        "polygon": dict(style="ring", polygon_xy_m=outline),
    }
    kinds = {}
    for name, kw in variants.items():
        worlds = []
        for model in ("lumped", "distributed"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                spec = FoundationElectrodeSpec(
                    size_xy_m=(10.0, 6.0),
                    orientation_deg=angle,
                    concrete_rho_ohm_m=rho_c,
                    concrete_model=model,
                    **kw,
                )
            worlds.append(_site(spec))
        assert gf.world_fingerprint(worlds[0]) == gf.world_fingerprint(worlds[1]), name
        kinds[name] = [e.kind for e in _owners(worlds[0])]
    assert kinds["ring"] == kinds["mesh 1 x 1"] == ["strip"] * 4
    assert kinds["mesh 1 x 2"] == ["strip"] * 3 + ["polyline"] * 2
    assert kinds["mesh 2 x 2"] == ["polyline"] * 6
    assert kinds["polygon"] == ["polyline"]
    description = FoundationElectrodeSpec.model_fields["concrete_model"].description
    for text in (_CONCRETE_MODEL_DEPRECATION, description):
        flat = " ".join(text.split())
        assert "no effect" in flat, flat
        assert "rectangle without cross-braces" in flat, flat
        assert "n_x = n_y = 1" in flat, flat


# user decision Q21
@pytest.mark.finding("10.12")
def test_model_copy_that_sets_the_deprecated_value_warns() -> None:
    """Pydantic does not validate ``model_copy(update=...)``; the spec checks
    the deprecated field itself, as the engine does for its flat fields.
    A copy that leaves the field alone stays silent."""
    with pytest.warns(DeprecationWarning, match="concrete_model") as caught:
        copied = _ring().model_copy(update={"concrete_model": "distributed"})
    assert [w.filename for w in caught if "concrete_model" in str(w.message)] == [
        __file__
    ]
    assert copied.concrete_model == "distributed"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        again = copied.model_copy(update={"depth_m": 1.0}, deep=True)
        _ring().model_copy(update={"concrete_model": "lumped"})
    assert _concrete_model_warnings(caught) == []
    assert again.concrete_model == "distributed" and again.depth_m == 1.0


# audit D2B-04
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("value", ["Lumped", "foo", None, 1])
@pytest.mark.parametrize("deep", [False, True])
def test_model_copy_rejects_an_invalid_concrete_model(
    value: object, deep: bool
) -> None:
    """Pydantic stores ``model_copy(update=...)`` unchecked. Up to the
    audit a value outside {"lumped", "distributed"} emitted the
    *deprecation* warning, was stored as it was, built four strips (any
    value but "lumped" did) and made the copy's JSON unloadable. It raises
    like the constructor now, without a warning, and the receiver keeps its
    value."""
    spec = _ring()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(pydantic.ValidationError) as copied:
            spec.model_copy(update={"concrete_model": value}, deep=deep)
        with pytest.raises(pydantic.ValidationError) as constructed:
            _ring(concrete_model=value)
    assert _concrete_model_warnings(caught) == []

    def _summary(error: pydantic.ValidationError) -> list[tuple]:
        return [(e["type"], e["loc"], e["msg"]) for e in error.errors()]

    assert _summary(copied.value) == _summary(constructed.value)
    assert copied.value.title == constructed.value.title == "FoundationElectrodeSpec"
    assert spec.concrete_model == "lumped"


# audit D2B-04
@pytest.mark.finding("10.12")
def test_model_copy_stores_only_valid_concrete_model_values() -> None:
    """Both valid values pass, "distributed" with one warning at the
    caller's line; the copy round-trips through JSON."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lumped = _ring().model_copy(update={"concrete_model": "lumped"})
    assert _concrete_model_warnings(caught) == []
    with pytest.warns(DeprecationWarning, match="concrete_model") as caught:
        distributed = _ring().model_copy(update={"concrete_model": "distributed"})
    assert [w.filename for w in caught if "concrete_model" in str(w.message)] == [
        __file__
    ]
    for copy in (lumped, distributed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            restored = FoundationElectrodeSpec.model_validate_json(
                copy.model_dump_json()
            )
        assert restored == copy


class _TaggedFoundation(FoundationElectrodeSpec):
    """A user's subclass with a required field of its own."""

    tag: str


def _errors(error: pydantic.ValidationError) -> list[tuple]:
    return [(e["type"], e["loc"], e["msg"]) for e in error.errors()]


# audit R2B-05
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("value", ["lumped", "distributed"])
@pytest.mark.parametrize("deep", [False, True])
def test_model_copy_of_a_subclass_with_a_required_field(value: str, deep: bool) -> None:
    """The D2B-04 fix validated the value by building a default instance of
    ``type(self)``; a subclass with a required field cannot be built from
    ``{"concrete_model": ...}`` alone, so a valid update raised
    ``ValidationError`` ("tag: Field required", for "distributed" after the
    deprecation warning). 0.15.0 returned the copy. Only the field is
    validated now: the copy keeps every other field, and "distributed" warns
    once, at the caller's line."""
    spec = _TaggedFoundation(tag="house-1", size_xy_m=(10.0, 6.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        copied = spec.model_copy(update={"concrete_model": value}, deep=deep)
    assert type(copied) is _TaggedFoundation
    assert (copied.tag, copied.size_xy_m, copied.concrete_model) == (
        "house-1",
        (10.0, 6.0),
        value,
    )
    files = [w.filename for w in caught if "concrete_model" in str(w.message)]
    assert files == ([__file__] if value == "distributed" else [])
    assert spec.concrete_model == "lumped"


# audit R2B-05
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("value", ["Lumped", None])
def test_model_copy_of_a_subclass_rejects_an_invalid_value(value: object) -> None:
    """An invalid value raises the subclass constructor's error, and only
    that error (the D2B-04 fix added "tag: Field required")."""
    spec = _TaggedFoundation(tag="house-1")
    with pytest.raises(pydantic.ValidationError) as copied:
        spec.model_copy(update={"concrete_model": value})
    with pytest.raises(pydantic.ValidationError) as constructed:
        _TaggedFoundation(tag="house-1", concrete_model=value)
    assert _errors(copied.value) == _errors(constructed.value)
    assert copied.value.title == constructed.value.title == "_TaggedFoundation"
    assert spec.concrete_model == "lumped"


# audit R2B-06
@pytest.mark.finding("10.12")
@pytest.mark.skipif(sys.version_info < (3, 13), reason="copy.replace needs Python 3.13")
def test_copy_replace_warns_at_the_callers_line() -> None:
    """``copy.replace`` (Python >= 3.13) reaches :meth:`model_copy` through
    pydantic's ``__replace__``; the warning was attributed to the standard
    library's ``copy.py``, and the default filters, which show a
    ``DeprecationWarning`` only for ``__main__``, hid it."""
    import copy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replaced = copy.replace(_ring(), concrete_model="distributed")
    files = [w.filename for w in caught if "concrete_model" in str(w.message)]
    assert files == [__file__]
    assert replaced.concrete_model == "distributed"


# audit D2B-06
@pytest.mark.finding("10.12")
def test_the_silent_paths_of_the_deprecation_are_documented() -> None:
    """Assigning the attribute, unpickling and ``model_construct`` bypass
    pydantic's validation, so they cannot warn; stream d2gen listed this as
    a known limitation, but no page said so. The deprecations page names
    the paths that warn and those that do not."""
    spec = _ring()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.concrete_model = "distributed"  # attribute assignment
        restored = pickle.loads(pickle.dumps(spec))  # noqa: S301
        constructed = FoundationElectrodeSpec.model_construct(
            concrete_model="distributed"
        )
    assert _concrete_model_warnings(caught) == []
    assert restored.concrete_model == constructed.concrete_model == "distributed"
    page = pathlib.Path(__file__).parents[1] / "docs" / "reference" / "deprecations.md"
    row = next(
        line
        for line in page.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `FoundationElectrodeSpec.concrete_model`")
    )
    for part in (
        "Assigning the attribute",
        "unpickling",
        "model_construct",
        "do not validate and do not warn",
        "`model_copy` whose `update` sets it",
    ):
        assert part in row, part


# user decision Q21
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("how", ["omitted", "keyword", "model_validate", "json"])
def test_default_concrete_model_is_accepted_silently(how: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if how == "omitted":
            spec = _ring()
        elif how == "keyword":
            spec = _ring(concrete_model="lumped")
        elif how == "model_validate":
            spec = FoundationElectrodeSpec.model_validate(
                {"style": "ring", "concrete_model": "lumped"}
            )
        else:
            spec = FoundationElectrodeSpec.model_validate_json(
                _ring().model_dump_json()
            )
    assert _concrete_model_warnings(caught) == []
    assert spec.concrete_model == "lumped"


# user decision Q21
@pytest.mark.finding("10.12")
def test_v0150_configuration_and_pickles_load_without_a_concrete_model_warning() -> (
    None
):
    """Configurations and pickles written by 0.15.0 carry
    ``"concrete_model": "lumped"`` in every foundation spec."""
    from groundfield.generators import TnNetworkConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TnNetworkConfig.model_validate_json(V0150_CONFIG.read_text())
        pickled = pickle.loads(V0150_PICKLES.read_bytes())  # noqa: S301
    assert _concrete_model_warnings(caught) == []
    foundations = [
        e
        for building in cfg.building_types
        for e in building.grounding.electrodes
        if e.kind == "foundation"
    ]
    assert len(foundations) == 4
    assert {e.concrete_model for e in foundations} == {"lumped"}
    assert pickled  # the fixture of tests/test_generator_pickles.py


# user decision Q21
@pytest.mark.finding("10.12")
def test_generator_paths_do_not_repeat_the_deprecation_warning() -> None:
    """The warning belongs to the line that sets the value. Resolving,
    sampling, planning, materialising, footprints, the layout builder and
    copies pass the spec on without warning again."""
    from groundfield.generators import (
        BuildingTypeSpec,
        LvNetworkLayout,
        TnNetworkConfig,
        TnNetworkGenerator,
        materialise_plan,
    )

    spec = _distributed_ring(concrete_rho_ohm_m=150.0, size_xy_m=(10.0, 8.0))
    grounding = GroundingSystemSpec(electrodes=[spec])
    footprint = gf.BuildingFootprint(
        polygon_xy_m=[(15.0, 10.0), (27.0, 10.0), (27.0, 19.0), (15.0, 19.0)], osm_id=7
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = TnNetworkConfig(
            building_types=[BuildingTypeSpec(name="house", grounding=grounding)],
            building_counts={"house": 2},
        )
        generator = TnNetworkGenerator(cfg, seed=0)
        plan = generator.plan()
        world = materialise_plan(plan)
        generator.build()
        generator.sample_world(1)
        cfg.sample(2)
        grounding.with_footprint(footprint, outline="ombr")
        grounding.with_footprint(footprint, outline="polygon")
        layout = LvNetworkLayout.from_footprints([footprint], substation_xy=(0.0, 0.0))
        layout.add_pen_cable(start="substation", end_xy=(40.0, 0.0))
        layout.connect_buildings()
        layout.to_world(seed=0, house_grounding=grounding)
        copied = pickle.loads(pickle.dumps(plan))
        cfg.model_copy(deep=True)
    assert _concrete_model_warnings(caught) == []
    assert copied == plan
    owners = [e for e in world.electrodes if "foundation" in e.name]
    assert [e.kind for e in owners] == ["strip"] * 8  # two houses, four sides


# user decision Q21
@pytest.mark.finding("10.12")
def test_plan_and_configuration_json_keep_the_deprecated_value() -> None:
    """A file that says ``"distributed"`` loads with the warning and builds
    the same world."""
    from groundfield.generators import (
        BuildingTypeSpec,
        NetworkPlan,
        TnNetworkConfig,
        TnNetworkGenerator,
        materialise_plan,
    )

    grounding = GroundingSystemSpec(electrodes=[_distributed_ring()])
    cfg = TnNetworkConfig(
        building_types=[BuildingTypeSpec(name="house", grounding=grounding)],
        building_counts={"house": 1},
    )
    plan = TnNetworkGenerator(cfg, seed=0).plan()
    with pytest.warns(DeprecationWarning, match="concrete_model"):
        restored_plan = NetworkPlan.model_validate_json(plan.model_dump_json())
    with pytest.warns(DeprecationWarning, match="concrete_model"):
        restored_cfg = TnNetworkConfig.model_validate_json(cfg.model_dump_json())
    assert restored_plan == plan
    assert restored_cfg == cfg
    assert gf.world_fingerprint(
        materialise_plan(restored_plan)
    ) == gf.world_fingerprint(materialise_plan(plan))


# audit D3A-01
@pytest.mark.finding("10.39")
@pytest.mark.parametrize(
    "size",
    [dict(size_xy_m=(10.0, -1.0)), dict(size_m=-10.0), dict(size_xy_m=(0.0, 5.0))],
    ids=["xy", "square", "zero"],
)
@pytest.mark.parametrize("style", ["ring", "mesh"])
def test_a_non_positive_side_is_rejected(size: dict, style: str) -> None:
    """Up to user decision Q38 the grid mesh of the default ring refused a
    non-positive side; four strips accepted a negative one and built the
    mirrored rectangle silently (10 m x -1 m: 9.47312 ohm like 10 m x 1 m;
    size_m=-10: 5.24988 ohm like 10 m). The spec rejects it now."""
    with pytest.raises(pydantic.ValidationError, match="greater than 0"):
        FoundationElectrodeSpec(style=style, n_x=1, n_y=1, depth_m=0.8, **size)


# audit D3A-01
@pytest.mark.finding("10.39")
def test_a_drawn_negative_side_is_rejected_when_the_site_is_built() -> None:
    """``size_m=Normal(2, 3)`` with seed 5 draws -0.41 m: four strips built a
    0.41 m ring (the grid mesh of the base before Q38 raised)."""
    spec = FoundationElectrodeSpec(
        style="ring", size_m=gf.generators.Normal(mean=2.0, std=3.0), depth_m=0.8
    )
    world = gf.World(soil=gf.HomogeneousSoil(rho=RHO))
    with pytest.raises(ValueError, match="sides of the rectangle must be positive"):
        GroundingSystemSpec(electrodes=[spec]).build_at(
            world, (0.0, 0.0), "h", np.random.default_rng(5)
        )


def _flat_doc(path: str) -> str:
    root = pathlib.Path(__file__).resolve().parents[1]
    return " ".join((root / path).read_text(encoding="utf-8").split())


# audit D3B-01
@pytest.mark.finding("10.39")
def test_one_owner_per_side_against_one_current_per_segment_holds_in_the_scope() -> (
    None
):
    """The texts said one owner per side is at most 0.9 % above one current
    per segment (short side >= 1 m); 150 m x 3 m (aspect ratio 50, inside
    the scope), bare, 5 mm at 0.8 m in 30 over 200 ohm m (h1 = 5 m) is
    +1.336 % (groundfield's per-segment assembly at 1 m segments; the
    independent boundary-element solution of audit D3B gives the same to
    0.001 points). The texts quote "at most 1.4 %" and the scope table."""
    dx, dy, ds = 150.0, 3.0, 1.0
    corners = [
        (-dx / 2, -dy / 2),
        (dx / 2, -dy / 2),
        (dx / 2, dy / 2),
        (-dx / 2, dy / 2),
    ]
    world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=200.0, h_1=5.0))
    names, side, lengths = [], [], []
    for k in range(4):
        (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
        n = max(1, round(math.hypot(x1 - x0, y1 - y0) / ds))
        for m in range(n):
            a = (x0 + (x1 - x0) * m / n, y0 + (y1 - y0) * m / n, 0.8)
            b = (x0 + (x1 - x0) * (m + 1) / n, y0 + (y1 - y0) * (m + 1) / n, 0.8)
            gf.create_electrode(
                world, "strip", name=f"s{len(names)}", start=a, end=b, wire_radius=0.005
            )
            names.append(f"s{len(names)}")
            side.append(k)
            lengths.append(math.dist(a, b))
    engine = gf.create_engine(frequencies=[50.0], segment_length=ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.asarray(gf.solve_mutual_matrix(world, engine, names)).real
    lengths, side = np.asarray(lengths), np.asarray(side)
    l_side = np.array([lengths[side == k].sum() for k in range(4)])
    t = np.zeros((len(names), 4))
    t[np.arange(len(names)), side] = lengths / l_side[side]

    def z_in(z: np.ndarray) -> float:
        one = np.ones(z.shape[0])
        return float(1.0 / (one @ np.linalg.solve(z, one)))

    per_side = 100.0 * (z_in(t.T @ p @ t) / z_in(p) - 1.0)
    assert per_side == pytest.approx(1.336, abs=0.002)
    assert "at most 1.4 % high over that scope" in " ".join(
        materialise_module_doc().split()
    )
    page = _flat_doc("docs/explanation/concrete-shells.md")
    assert "one owner per side is at most 1.4 % high" in page
    assert "+1.336 % (150 m × 3 m, same soil)" in page
    adr = _flat_doc("docs/adr/0012-foundation-concrete-encasement.md")
    assert "at most 1.4 % high, 150 m × 3 m bare" in adr


def materialise_module_doc() -> str:
    from groundfield.generators import materialise

    return materialise.__doc__ or ""


# audit D3B-02
@pytest.mark.finding("10.39")
def test_the_texts_quote_the_field_change_at_the_passive_foundation() -> None:
    """concrete-shells.md and ADR-0012 said the field next to the passive
    foundation of ``mutual_foundations_lumped_concrete`` moves by 1.6 %: the
    golden harness's max |change| / max |R|. At the edges of the passive
    foundation it moves by +5.49 / -4.70 % with 1 m segments (both kernels;
    the default line kernel gives the same at 2 m) and by +5.94 / -5.11 %
    with the golden engine's 2 m segments and the point kernel; the texts
    quoted the latter as the change (audit R4B-13: a segmentation effect)."""

    from groundfield.solver._discretize import _one_owner_per_wire_off

    def pair(one_owner: bool, segment_length: float) -> np.ndarray:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=100.0, h_1=5.0))
        c = gf.geometry.electrodes.concrete_shell_coefficient(100.0, 0.0075, 0.05)
        anchors = []
        for k, x in enumerate((0.0, 30.0)):
            if one_owner:
                name = f"h{k}_foundation_0"
                gf.create_electrode(
                    w,
                    "grid_mesh",
                    name=name,
                    corner=(x - 5.0, -5.0, 0.7),
                    size=(10.0, 10.0),
                    n_x=1,
                    n_y=1,
                    wire_radius=0.0575,
                    concrete_shell_coefficient_ohm_m=c,
                )
                anchors.append(name)
            else:
                spec = GroundingSystemSpec(
                    electrodes=[
                        FoundationElectrodeSpec(
                            style="ring",
                            size_m=10.0,
                            depth_m=0.7,
                            wire_radius_m=0.0075,
                            concrete_rho_ohm_m=100.0,
                            concrete_thickness_m=0.05,
                        )
                    ],
                    bond_conductor_type="bare_copper",
                )
                anchors.append(
                    spec.build_at(w, (x, 0.0), f"h{k}", np.random.default_rng(1))
                )
        engine = gf.create_engine(
            backend="image",
            kernel="point",
            segment_length=segment_length,
            image_max_terms=500,
            frequencies=[50.0],
        )
        points = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        # the one-owner grid of the time (user decision Q30 split it by wire)
        with warnings.catch_warnings(), _one_owner_per_wire_off():
            warnings.simplefilter("ignore")
            return np.asarray(gf.solve_mutual_field(w, engine, anchors, points)).real

    for segment_length, expected in ((2.0, [5.94, -5.11]), (1.0, [5.49, -4.70])):
        change = 100.0 * (
            pair(False, segment_length)[:, 1] / pair(True, segment_length)[:, 1] - 1.0
        )
        assert change == pytest.approx(expected, abs=0.01)
    for path in (
        "docs/explanation/concrete-shells.md",
        "docs/adr/0012-foundation-concrete-encasement.md",
    ):
        text = _flat_doc(path)
        assert "+5.5 / −4.7 %" in text and "+5.9 / −5.1 %" in text
        assert "the field next to the passive foundation by 1.6 %" not in text
        assert "golden engine, point kernel; +5.5" not in text


# audits R4B-05, R4B-06
@pytest.mark.finding("10.39")
def test_the_texts_quote_one_owner_up_to_2_2_percent_and_the_scanned_rectangle() -> (
    None
):
    """The texts quoted one owner "up to 2.1 %" over the scope next to
    "2.16 % (16 m x 1 m)" in the same sentence (groundfield's per-segment
    assembly +2.106 / +2.160 % at 0.5 / 0.25 m segments, audit R4B's BEM
    +2.103 / +2.153 %), and "150 m x 2.9 m: +1.335 %" for the scanned
    145 m x 2.9 m (150 m x 2.9 m is aspect 51.7, outside the scope: +1.350 %)."""
    for path in (
        "docs/explanation/concrete-shells.md",
        "docs/adr/0012-foundation-concrete-encasement.md",
    ):
        text = _flat_doc(path)
        assert "up to 2.1 %" not in text and "up to 2.2 %" in text, path
    text = _flat_doc("docs/explanation/concrete-shells.md")
    assert "150 m × 2.9 m: +1.335 %" not in text
    assert "145 m × 2.9 m: +1.335 %" in text
    from groundfield.generators import FoundationElectrodeSpec as spec
    from groundfield.generators import materialise

    for doc in (materialise.__doc__, spec.model_fields["concrete_model"].description):
        assert "2.1 %" not in " ".join((doc or "").split())


# audit W1A-04 (user decisions Q30, Q44)
_STALE_OWNER_PHRASES = (
    r"uniform leakage current per electrode",
    r"\bis one closed wire\b",
    r"one closed polyline,? one owner\b",
    r"one closed polyline \(one owner\)",
    r"\boutline (\(`{1,2}polygon_xy_m`{1,2}\) )?is one closed polyline\s*[.;]",
)


@pytest.mark.finding("10.22")
def test_no_text_describes_the_owners_of_before_q30_and_q44() -> None:
    """Since user decisions Q30 and Q44 the image family solves one current
    per straight wire, and a building outline is one owner per straight side.
    The source and the documentation (outside the audit records and the
    release notes) no longer say one uniform current per electrode or one
    closed wire for an outline (audit W1A-04: ``electrode_specs.py``,
    ``materialise.py`` and the generators API page still did)."""
    root = pathlib.Path(gf.__file__).resolve().parents[2]
    files = [
        *sorted((root / "src").rglob("*.py")),
        *sorted(p for p in (root / "docs").rglob("*.md") if "audit" not in p.parts),
    ]
    assert len(files) > 100
    hits = []
    for path in files:
        text = " ".join(path.read_text(encoding="utf-8").split())
        for phrase in _STALE_OWNER_PHRASES:
            for match in re.finditer(phrase, text):
                hits.append(f"{path.relative_to(root)}: {match.group(0)!r}")
    assert not hits, hits
