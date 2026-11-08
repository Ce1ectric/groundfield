r"""Uniform-current owners of generated foundation electrodes.

The image family solves for one uniform leakage current per electrode (an
"owner"). Up to 2026.11 the generator built an axis-aligned bare mesh
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

Rule since 2026.11 (one owner set per structure, independent of the
orientation, of the way the outline is described and of whether a shell is
configured):

* a mesh foundation (cross-braces) has one owner per wire line;
* a perimeter-only foundation (``style="ring"`` or a ``polygon_xy_m``
  outline) has one owner for the closed perimeter with
  ``concrete_model="lumped"`` and one owner per straight wire (side or
  polygon edge) with ``"distributed"``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import FoundationElectrodeSpec, GroundingSystemSpec

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


# audit A2G-04
@pytest.mark.finding("10.12")
@pytest.mark.parametrize("thickness_m", [0.05, 1e-4], ids=["t=5cm", "t=0.1mm"])
def test_vanishing_shell_on_a_distributed_ring_is_continuous(
    thickness_m: float,
) -> None:
    """A bare ``"distributed"`` ring was one owner, an encased one four:
    -0.24 % for a 20 x 2 m ring."""
    bare = _z(_site(_ring(concrete_model="distributed")), "point")
    shell = _z(
        _site(
            _ring(
                concrete_model="distributed",
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


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("rho_c", [None, 1000.0])
def test_ring_foundation_does_not_depend_on_its_orientation(
    model: str, rho_c: float | None
) -> None:
    """A lumped ring was one owner at 0 deg and four at 30 deg: +0.24 % for a
    20 x 2 m ring (0.30 % at aspect ratio 50)."""
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


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("angle", [0.0, 30.0])
def test_rectangle_as_polygon_equals_the_rectangle_spec(
    model: str, angle: float
) -> None:
    """The same rectangle described by ``polygon_xy_m`` or by
    ``size_xy_m`` / ``orientation_deg`` is the same model (distributed: one
    owner per side, also for the polygon)."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    corners = [(-10.0, -1.0), (10.0, -1.0), (10.0, 1.0), (-10.0, 1.0)]
    polygon = tuple((c * x - s * y, s * x + c * y) for x, y in corners)
    shell = {"concrete_rho_ohm_m": 500.0, "concrete_model": model}
    z_polygon = _z(_site(_ring(polygon_xy_m=polygon, **shell)))
    z_spec = _z(_site(_ring(orientation_deg=angle, **shell)))
    assert z_polygon == pytest.approx(z_spec, rel=1e-6)


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


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("angle", [None, 30.0])
@pytest.mark.parametrize("rho_c", [None, 150.0])
@pytest.mark.parametrize(
    ("model", "kinds"),
    [("lumped", None), ("distributed", ["strip"] * 4)],
)
def test_ring_owners_follow_the_concrete_model_only(
    angle: float | None, rho_c: float | None, model: str, kinds: list | None
) -> None:
    world = _site(
        _ring(orientation_deg=angle, concrete_rho_ohm_m=rho_c, concrete_model=model)
    )
    owners = _owners(world)
    if kinds is None:  # lumped: one owner for the closed perimeter
        expected = ["grid_mesh"] if angle is None else ["polyline"]
        assert [e.kind for e in owners] == expected
    else:
        assert [e.kind for e in owners] == kinds
    assert sum(e.total_length() for e in owners) == pytest.approx(44.0)


# audit A2G-04
@pytest.mark.finding("10.39")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_polygon_owners_follow_the_concrete_model(model: str) -> None:
    outline = (
        (-6.0, -4.5),
        (6.0, -4.5),
        (6.0, 4.5),
        (-1.0, 4.5),
        (-1.0, 1.5),
        (-6.0, 1.5),
    )
    world = _site(_ring(polygon_xy_m=outline, concrete_model=model))
    owners = _owners(world)
    if model == "lumped":
        assert [e.kind for e in owners] == ["polyline"] and owners[0].closed
    else:
        assert [e.kind for e in owners] == ["strip"] * 6
    assert sum(e.total_length() for e in owners) == pytest.approx(42.0)


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
