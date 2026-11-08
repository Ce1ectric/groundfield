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
* a ring foundation has one owner per side with ``"distributed"`` at every
  angle; with ``"lumped"`` one ``grid_mesh`` owner when axis-aligned and
  one owner per side when rotated (at most 0.06 % apart up to aspect
  ratio 4, 0.24 % at 10 for a bare ring in homogeneous soil);
* a ``polygon_xy_m`` outline is one closed polyline for both values.

Since that choice is all ``concrete_model`` still does, the field is
deprecated (user decision Q21): ``"distributed"`` warns at the user's line,
the default and the files of 0.15.0 load silently, and the generators do
not repeat the warning.
"""

from __future__ import annotations

import pathlib
import pickle
import warnings

import numpy as np
import pydantic
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
        "style='ring'",
        "one per side",
        "0.16 %",
        "0.67 %",
        "0.06 %",
        "0.3 %",
        "2027.5",
        "drop the argument",
    ):
        assert part in message, part
    assert spec.concrete_model == "distributed"
    world = _site(spec)
    assert [e.kind for e in _owners(world)] == ["strip"] * 4


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
