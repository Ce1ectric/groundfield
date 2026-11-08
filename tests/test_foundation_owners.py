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

Rule since 2026.11 (the owners never depend on whether a shell is
configured):

* a mesh foundation (cross-braces) has one owner per wire line at every
  angle, for both ``concrete_model`` values;
* a rectangle without cross-braces (``style="ring"``, or ``"mesh"`` with
  ``n_x = n_y = 1``) has one owner per side with ``"distributed"`` at
  every angle; with ``"lumped"`` one ``grid_mesh`` owner when axis-aligned
  and one owner per side when rotated (less than 1 % apart up to aspect
  ratio 4 and 2 % up to 50 over the AP1 soils for a short side of at least
  1 m, bare or in concrete up to 0.2 m thick; audit D2B-02, R2B-01);
* a ``polygon_xy_m`` outline is one closed polyline for both values.

Since that choice is all ``concrete_model`` still does, the field is
deprecated (user decision Q21): ``"distributed"`` warns at the user's line,
the default and the files of 0.15.0 load silently, and the generators do
not repeat the warning.
"""

from __future__ import annotations

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
@pytest.mark.parametrize("rho_c", [None, 1000.0])
def test_distributed_ring_does_not_depend_on_its_orientation(
    rho_c: float | None,
) -> None:
    """A bare distributed ring was one owner at 0 deg and four at 30 deg."""
    values = [
        _z(
            _site(
                _ring(
                    orientation_deg=angle,
                    concrete_model="distributed",
                    concrete_rho_ohm_m=rho_c,
                )
            )
        )
        for angle in (None, 30.0, 90.0)
    ]
    assert values == pytest.approx([values[0]] * 3, rel=1e-6)


# audit A2G-04
@pytest.mark.finding("10.39")
def test_lumped_ring_keeps_its_axis_aligned_single_owner() -> None:
    """Documented exception: a rotated lumped ring stays one owner per side
    (a closed polyline would drop the interior sample of the default
    touch-voltage envelope, which decides it for rings of 20 m and more).
    The difference to the single axis-aligned owner is 0.24 % for this
    20 x 2 m ring (aspect ratio 10)."""
    z_axis = _z(_site(_ring()))
    z_turned = _z(_site(_ring(orientation_deg=30.0)))
    assert 0.0 < z_axis / z_turned - 1.0 < 3e-3


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
@pytest.mark.parametrize("model", ["lumped", "distributed"])
def test_ring_owners(angle: float | None, rho_c: float | None, model: str) -> None:
    world = _site(
        _ring(orientation_deg=angle, concrete_rho_ohm_m=rho_c, concrete_model=model)
    )
    owners = _owners(world)
    if model == "lumped" and angle is None:
        assert [e.kind for e in owners] == ["grid_mesh"]
    else:
        assert [e.kind for e in owners] == ["strip"] * 4
    assert sum(e.total_length() for e in owners) == pytest.approx(44.0)


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
# concrete_model is deprecated (user decision Q21)
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


# user decision Q21
@pytest.mark.finding("10.12")
def test_distributed_concrete_model_warns_at_the_callers_line() -> None:
    """The non-default value keeps working and warns: what the field still
    does, how much it changes, the earliest removal, what to do."""
    with pytest.warns(DeprecationWarning, match="concrete_model") as caught:
        spec = _ring(concrete_model="distributed")
    (warning,) = [w for w in caught if "concrete_model" in str(w.message)]
    assert warning.filename == __file__  # the user's line, not groundfield's
    message = str(warning.message)
    for part in (
        # audit D2B-05: every foundation whose owners the value changes
        "rectangle without cross-braces",
        "style='ring'",
        "n_x = n_y = 1",
        "one per side",
        # audit D2B-02, R2B-01: the bound with its scope
        "less than 1 % up to aspect ratio 4 (short side at least 1 m, sides up to 80 m)",
        "2 % up to aspect ratio 50 (short side at least 1 m, sides up to 150 m)",
        "up to 0.2 m thick",
        "0.4 % and 1 %",
        "2027.5",
        "drop the argument",
    ):
        assert part in message, part
    assert spec.concrete_model == "distributed"
    world = _site(spec)
    assert [e.kind for e in _owners(world)] == ["strip"] * 4


def _quoted_bounds(text: str) -> dict[str, float]:
    """The envelope a text quotes: all shells and the default shell."""
    text = " ".join(text.split())
    found = re.search(
        r"less than ([0-9.]+) % up to aspect ratio 4 .*?and ([0-9.]+) % up to "
        r"aspect ratio 50 .*?by less than ([0-9.]+) % and ([0-9.]+) %",
        text,
    )
    assert found, text
    return {
        "4": float(found.group(1)),
        "50": float(found.group(2)),
        "4-default": float(found.group(3)),
        "50-default": float(found.group(4)),
    }


def _lumped_over_distributed(
    size_xy_m: tuple[float, float],
    soil: object,
    rho_c: float | None,
    thickness_m: float,
    style: str = "ring",
) -> float:
    """Z(lumped) / Z(distributed) - 1 of the generated rectangle (default
    engine, 0.8 m, 5 mm wire)."""
    values = []
    for model in ("lumped", "distributed"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = FoundationElectrodeSpec(
                style=style,
                size_xy_m=size_xy_m,
                n_x=1,
                n_y=1,
                depth_m=0.8,
                wire_radius_m=0.005,
                concrete_rho_ohm_m=rho_c,
                concrete_thickness_m=thickness_m,
                concrete_model=model,
            )
        world = gf.World(soil=soil)
        anchor = GroundingSystemSpec(electrodes=[spec]).build_at(
            world, (0.0, 0.0), "h", np.random.default_rng(0)
        )
        world.add_source(gf.CurrentSource(name="s", attached_to=anchor, magnitude=1.0))
        engine = gf.create_engine(frequencies=[50.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values.append(complex(engine.solve(world).input_impedance()[0]).real)
    return values[0] / values[1] - 1.0


# audit D2B-02, audit R2B-01, audit R2B-02
@pytest.mark.finding("10.12")
@pytest.mark.parametrize(
    ("size_xy_m", "soil", "rho_c", "thickness_m", "bound", "measured"),
    [
        # the largest value found in each clause of the scope (evidence f3gen
        # r2b_scope_scan.py, r2b_scope_refine.py, r2b_scope_refine_analyse.out)
        ((4.0, 1.0), (1000.0, 30.0, 5.0), 30.0, 0.2, "4", 0.006906),
        ((16.0, 1.0), (1000.0, 30.0, 10.0), 30.0, 0.2, "50", 0.015531),
        ((80.0, 20.0), (30.0, 200.0, 5.0), 30.0, 0.05, "4-default", 0.003207),
        ((150.0, 7.75), (30.0, 200.0, 5.0), None, 0.05, "50-default", 0.009150),
        ((7.0, 1.75), (100.0, 100.0, None), None, 0.05, "4-default", 0.000887),
        ((25.0, 1.0), (100.0, 100.0, None), None, 0.05, "50-default", 0.003308),
        # ... and the cases with which the auditor broke the earlier bounds
        ((10.0, 2.5), (100.0, 100.0, None), None, 0.05, "4-default", 0.00084),
        ((40.0, 10.0), (30.0, 200.0, 5.0), None, 0.05, "4-default", 0.00253),
        ((50.0, 1.0), (1000.0, 30.0, 30.0), 30.0, 0.2, "50", 0.01241),
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
    ],
)
def test_the_quoted_concrete_model_bound_holds(
    size_xy_m: tuple[float, float],
    soil: tuple[float, float, float | None],
    rho_c: float | None,
    thickness_m: float,
    bound: str,
    measured: float,
) -> None:
    """The bounds of the deprecation (0.16 % up to aspect ratio 4, 0.67 % up
    to 50) held only for the six rings measured; the auditor found 0.38 % and
    1.24 %. The second bounds (0.7 % / 1.6 %; 0.33 % / 0.92 %) were scan
    maxima, not maxima of the scope (16 m x 1 m 1.553 %, 150 m x 7.75 m
    0.915 %, 25 m x 1 m bare in homogeneous soil 0.331 % against "0.33 %";
    audit R2B-02). The bound is quoted with a margin over the largest value
    found in each clause of the stated scope (short side at least 1 m,
    sides up to 80 m / 150 m, bare or 30 ... 1000 ohm m up to 0.2 m thick,
    the 39 AP1 soils, 0.8 m, 5 mm); the field description says the same,
    and a mesh without braces behaves like the ring (audit D2B-05)."""
    r1, r2, h = soil
    model = (
        gf.HomogeneousSoil(rho=r1)
        if h is None
        else gf.TwoLayerSoil(rho_1=r1, rho_2=r2, h_1=h)
    )
    diff = _lumped_over_distributed(size_xy_m, model, rho_c, thickness_m)
    assert diff == pytest.approx(measured, abs=5e-6)
    for text in (
        _CONCRETE_MODEL_DEPRECATION,
        FoundationElectrodeSpec.model_fields["concrete_model"].description,
    ):
        assert abs(diff) < _quoted_bounds(text)[bound] / 100.0
    mesh = _lumped_over_distributed(size_xy_m, model, rho_c, thickness_m, style="mesh")
    assert mesh == pytest.approx(diff, abs=1e-12)


_DOCS = pathlib.Path(__file__).parents[1] / "docs"
_SCOPE = re.compile(
    r"less than ([0-9.]+) % up to aspect ratio 4 \(short side at least ([0-9.]+) m, "
    r"sides up to 80 m\) and ([0-9.]+) % up to aspect ratio 50 \(short side at "
    r"least ([0-9.]+) m, sides up to 150 m\); bare or with the default 5 cm shell "
    r"by less than ([0-9.]+) % and ([0-9.]+) %"
)


def _texts_with_the_scope() -> dict[str, str]:
    """Every text that quotes the bound with its scope, whitespace folded."""
    from groundfield.generators import materialise

    def read(*parts: str) -> str:
        return _DOCS.joinpath(*parts).read_text(encoding="utf-8")

    deprecations = next(
        line
        for line in read("reference", "deprecations.md").splitlines()
        if line.startswith("| `FoundationElectrodeSpec.concrete_model`")
    )
    adr = read("adr", "0012-foundation-concrete-encasement.md")
    page = read("explanation", "concrete-shells.md")
    texts = {
        "warning": _CONCRETE_MODEL_DEPRECATION,
        "field description": FoundationElectrodeSpec.model_fields[
            "concrete_model"
        ].description,
        "materialise module": materialise.__doc__,
        "deprecations page": deprecations,
        "ADR-0012": adr.split("### User decision Q21", 1)[1],
        "concrete-shell page": page.split(
            "## Owners of a rectangle without cross-braces", 1
        )[1].split("\n## ", 1)[0],
    }
    return {name: " ".join(text.split()) for name, text in texts.items()}


# audit R2B-01, audit R2B-02
@pytest.mark.finding("10.12")
def test_every_text_quotes_the_bound_with_the_short_side_in_both_clauses() -> None:
    """The aspect-ratio-4 clause named no shortest side, but the scan behind
    it started at 4 m x 1 m: narrower rectangles exceed the bound (3.6 m x
    0.9 m 0.729 %, 2 m x 0.5 m 1.016 % with 0.2 m of wet concrete over
    1000 / 30 ohm m; 1.2 m x 0.3 m 0.331 % with the default shell). Every
    text now quotes one bound, with a short side of at least 1 m in both
    clauses (the materialise docstring also quoted "bare in homogeneous soil
    by less than 0.09 % and 0.33 %", broken by 25 m x 1 m at 0.331 %), and
    the short summaries give the same numbers."""
    quoted, problems = {}, []
    for name, text in _texts_with_the_scope().items():
        found = _SCOPE.search(text)
        if found is None:
            problems.append(f"{name}: no bound with the short side in both clauses")
        else:
            quoted[name] = tuple(float(g) for g in found.groups())
        if "bare in homogeneous soil by less than" in text:
            problems.append(f"{name}: a bare-homogeneous bound without margin")
    summary = (
        "less than 1 % apart up to aspect ratio 4 and 2 % up to 50 for a short "
        "side of at least 1 m"
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
    narrow = _lumped_over_distributed(
        (2.0, 0.5), gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0), 30.0, 0.2
    )
    assert narrow == pytest.approx(0.010158, abs=5e-6)
    assert narrow > quoted["warning"][0] / 100.0


# audit R2B-07
@pytest.mark.finding("10.12")
def test_the_owner_section_states_its_model_uncertainty() -> None:
    """The largest values of the bound sit at the 0.2 m shell, whose outer
    radius (0.205 m) makes segment length / wire radius 2.4 at the default
    0.5 m segments, below the thin-wire limit of 5 (the solve warns). There an
    independent solver that averages the potential over the wire surface
    gives 0.598 / 1.468 % instead of 0.691 / 1.553 %, and the comparison with
    one current per segment is not converged; the page said "an independent
    thin-wire boundary-element solution reproduces the largest cases to 0.01
    points" and stated no model uncertainty."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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
        assert "0.598" in text and "1.468" in text
    assert "segment length over wire radius is 2.4" in page
    assert "is not converged" in adr
    assert "refining its segments to 0.083 m" in page


# audit D2B-05
@pytest.mark.finding("10.12")
def test_the_texts_name_every_foundation_whose_owners_concrete_model_changes() -> None:
    """A ``style="mesh"`` foundation with ``n_x = n_y = 1`` is one grid mesh
    with "lumped" and four strips with "distributed", like a ring, but the
    warning and the field description spoke of ``style="ring"`` only and
    counted mesh foundations among those the value does not change."""

    def owners(model: str, **kw: object) -> list[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec = FoundationElectrodeSpec(
                size_xy_m=(10.0, 6.0), concrete_model=model, **kw
            )
        return [e.kind for e in _owners(_site(spec))]

    assert owners("lumped", style="mesh", n_x=1, n_y=1) == ["grid_mesh"]
    assert owners("distributed", style="mesh", n_x=1, n_y=1) == ["strip"] * 4
    # one brace is enough: one owner per wire line for both values
    braced = owners("lumped", style="mesh", n_x=1, n_y=2)
    assert braced == owners("distributed", style="mesh", n_x=1, n_y=2)
    assert braced == ["strip"] * 3 + ["polyline"] * 2
    description = FoundationElectrodeSpec.model_fields["concrete_model"].description
    for text in (_CONCRETE_MODEL_DEPRECATION, description):
        flat = " ".join(text.split())
        assert "rectangle without cross-braces" in flat, flat
        assert "n_x = n_y = 1" in flat, flat
    assert "mesh foundations (one owner per wire)" not in " ".join(description.split())


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
