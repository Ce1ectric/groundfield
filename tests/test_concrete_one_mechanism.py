r"""Concrete-encased foundations of generated worlds: one mechanism.

A foundation wire of radius :math:`r_a` in a concrete shell of
resistivity :math:`\rho_c` and thickness :math:`t` is seen by the soil
through the radius :math:`r_b = r_a + t`; the shell adds the per-length
coefficient

.. math::

   C = \frac{\rho_c}{2\pi}\ln\frac{r_b}{r_a}\quad[\Omega\,\mathrm{m}]

on every encased wire (ADR-0012). Up to v0.15.0 ``concrete_model="lumped"``
wrote :math:`C/L_\text{perim}` into ``World.concrete_shell_corrections``,
which only the TN generator's PEN service drops read: ``solve_mutual_*``
(the AP1 Z_G path), the layout builder and every substation or cabinet
foundation dropped the shell (findings 10.12, 10.14), and the service drop
sent the current of bonded rods through the foundation's shell (10.13).
Since 2026.11 both models put :math:`C` on the encased wires themselves.
Every wire of an encased foundation is encased, the cross-braces of a mesh
foundation included (user decision Q20: a foundation earth electrode per
DIN 18014 lies in the concrete with its connections); rods stay bare.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    CabinetConfig,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    HomogeneousSoilSpec,
    LvNetworkLayout,
    RodElectrodeSpec,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
)

R_A = 0.0075
T = 0.05
POLY = ((-6.0, -4.5), (6.0, -4.5), (6.0, 4.5), (-1.0, 4.5), (-1.0, 1.5), (-6.0, 1.5))


def _coefficient(rho_c: float, r_a: float = R_A, t: float = T) -> float:
    return rho_c / (2.0 * math.pi) * math.log((r_a + t) / r_a)


def _foundation(model: str, rho_c: float | None = 100.0, **kw: object):
    params = dict(
        style="ring",
        size_m=10.0,
        depth_m=0.7,
        wire_radius_m=R_A,
        concrete_rho_ohm_m=rho_c,
        concrete_thickness_m=T,
        concrete_model=model,
    )
    params.update(kw)
    return FoundationElectrodeSpec(**params)


def _engine(**kw: object) -> gf.Engine:
    return gf.create_engine(
        backend="image", segment_length=1.0, frequencies=[50.0], kernel="point", **kw
    )


def _pair_zg(model: str, rho_c: float | None, *, rho_1: float = 30.0, **kw: object):
    """AP1 recipe: two foundations built with build_at, Z_G by solve_mutual_matrix."""
    spec = GroundingSystemSpec(electrodes=[_foundation(model, rho_c, **kw)])
    world = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=100.0, h_1=5.0))
    rng = np.random.default_rng(1)
    anchors = [
        spec.build_at(world, site_xy=(0.0, 0.0), name_prefix="h0", rng=rng),
        spec.build_at(world, site_xy=(30.0, 0.0), name_prefix="h1", rng=rng),
    ]
    engine = gf.create_engine(
        backend="image", segment_length=2.0, image_max_terms=500, kernel="point"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # no registry entries
        z = gf.solve_mutual_matrix(world, engine, anchors)  # galerkin (Q15)
    return np.asarray(z), world


def _solve_site(electrodes: list, *, rho: float = 100.0) -> complex:
    """Cluster impedance of one site built with build_at (no PEN, no drop)."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=rho))
    anchor = GroundingSystemSpec(electrodes=electrodes).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="site", rng=np.random.default_rng(0)
    )
    gf.create_source(world, attached_to=anchor, magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return world.solve(_engine()).input_impedance()[0]


# ---------------------------------------------------------------------
# 10.12: the lumped shell reaches every solve
# ---------------------------------------------------------------------


@pytest.mark.finding("10.12")
def test_lumped_shell_is_a_coefficient_on_the_encased_wire() -> None:
    _, world = _pair_zg("lumped", 100.0, polygon_xy_m=POLY)
    assert world.concrete_shell_corrections == {}
    for electrode in world.electrodes:
        assert electrode.wire_radius == pytest.approx(R_A + T)
        assert electrode.concrete_shell_coefficient_ohm_m == pytest.approx(
            _coefficient(100.0), rel=1e-15
        )


@pytest.mark.finding("10.12")
@pytest.mark.parametrize("rho_1", [30.0, 100.0, 1000.0])
def test_lumped_polygon_foundation_equals_the_distributed_one(rho_1: float) -> None:
    """A polygon ring is one closed wire: both models are the same world.
    v0.15.0 dropped the lumped shell in ``solve_mutual_matrix``: Z11
    -28.6 / -15.2 / -2.7 % against the distributed model."""
    z_l, _ = _pair_zg("lumped", 100.0, rho_1=rho_1, polygon_xy_m=POLY)
    z_d, _ = _pair_zg("distributed", 100.0, rho_1=rho_1, polygon_xy_m=POLY)
    assert np.allclose(z_l, z_d, rtol=1e-12, atol=0.0)


@pytest.mark.finding("10.12")
def test_lumped_shell_resistivity_moves_the_mutual_matrix() -> None:
    """v0.15.0: rho_c = 100 and 5000 ohm m gave bit-identical Z_G."""
    z_100, _ = _pair_zg("lumped", 100.0)
    z_5000, _ = _pair_zg("lumped", 5000.0)
    # Diagonal: + (C(5000) - C(100)) / L_perim on each foundation.
    added = (_coefficient(5000.0) - _coefficient(100.0)) / 40.0
    assert z_5000[0, 0].real - z_100[0, 0].real == pytest.approx(added, rel=1e-3)


@pytest.mark.finding("10.12")
def test_lumped_and_distributed_square_ring_agree_within_adr_0012() -> None:
    """Axis-aligned ring: one grid-mesh owner (lumped) vs four strip owners
    (distributed); ADR-0012 asks for 2 %."""
    z_l, _ = _pair_zg("lumped", 1000.0)
    z_d, _ = _pair_zg("distributed", 1000.0)
    assert np.allclose(z_l, z_d, rtol=0.02)


@pytest.mark.finding("10.12")
def test_lumped_shell_adds_exactly_its_sunde_resistance() -> None:
    """Uniform current per electrode: Z(shell) - Z(bare wire of radius r_b)
    = C / L_perim."""
    bare_rb = _solve_site([_foundation("lumped", None, wire_radius_m=R_A + T)])
    shell = _solve_site([_foundation("lumped", 500.0)])
    assert (shell - bare_rb).real == pytest.approx(_coefficient(500.0) / 40.0, rel=1e-9)


# ---------------------------------------------------------------------
# 10.13: rods and braces bypass the shell; every site's shell counts
# ---------------------------------------------------------------------


@pytest.mark.finding("10.13")
@pytest.mark.parametrize("rho_c", [150.0, 1000.0, 5000.0])
def test_mixed_site_lumped_equals_distributed(rho_c: float) -> None:
    """Foundation + 2 m rod in a TN network: the rod's current does not
    pass through the foundation's shell. v0.15.0 (V1 on the service drop):
    lumped - distributed = +1.4 / +9.7 / +26.8 %."""

    def z_in(model: str) -> complex:
        building = BuildingTypeSpec(
            name="house",
            grounding=GroundingSystemSpec(
                electrodes=[
                    _foundation(model, rho_c, depth_m=0.8, wire_radius_m=0.005),
                    RodElectrodeSpec(length_m=2.0, offset_xy_m=(7.0, 0.0)),
                ]
            ),
        )
        cfg = TnNetworkConfig(
            soil=HomogeneousSoilSpec(rho=100.0),
            building_types=[building],
            building_counts={"house": 4},
        )
        world = TnNetworkGenerator(cfg, seed=0).build()
        return _engine().solve(world).input_impedance()[0]

    assert z_in("lumped").real == pytest.approx(z_in("distributed").real, rel=0.02)


def _mesh_wires(spec: FoundationElectrodeSpec, soil: object | None = None) -> list:
    """The wires of one mesh foundation built with build_at."""
    world = gf.create_world(
        soil=soil if soil is not None else gf.HomogeneousSoil(rho=100.0)
    )
    GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    return [e for e in world.electrodes if e.name.startswith("h_foundation")]


# user decision Q20
@pytest.mark.finding("10.13")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("orientation_deg", [None, 30.0])
def test_mesh_braces_are_encased(model: str, orientation_deg: float | None) -> None:
    """DIN 18014: a foundation earth electrode lies in the concrete with its
    connections, so the braces carry the shell like the perimeter. On the
    2026.11 development line before the user decision Q20 the two braces
    kept r_a and C = 0 (in 0.15.0 they had r_b and C = 0)."""
    spec = _foundation(model, 500.0, style="mesh", n_x=2, n_y=2)
    wires = _mesh_wires(spec.model_copy(update={"orientation_deg": orientation_deg}))
    assert len(wires) == 6  # 4 perimeter wires + 2 braces
    assert sum(e.total_length() for e in wires) == pytest.approx(60.0)
    for wire in wires:
        assert wire.wire_radius == pytest.approx(R_A + T)
        assert wire.concrete_shell_coefficient_ohm_m == pytest.approx(
            _coefficient(500.0), rel=1e-15
        )


@pytest.mark.finding("10.13")
@pytest.mark.parametrize("orientation_deg", [None, 30.0])
def test_bare_mesh_wires_keep_the_wire_radius(orientation_deg: float | None) -> None:
    """Without a shell nothing changes: every wire has r_a and C = 0."""
    spec = _foundation("lumped", None, style="mesh", n_x=3, n_y=2)
    wires = _mesh_wires(spec.model_copy(update={"orientation_deg": orientation_deg}))
    assert len(wires) == 7
    assert all(e.wire_radius == R_A for e in wires)
    assert all(e.concrete_shell_coefficient_ohm_m == 0.0 for e in wires)


# user decision Q20
@pytest.mark.finding("10.13")
def test_shell_of_a_mesh_foundation_spans_every_wire() -> None:
    """The lumped shell resistance of the whole foundation is C over the
    whole encased length: the per-wire C / L_j in parallel give C / 60 m for
    a 10 m 2 x 2 mesh (C / 40 m, the perimeter, on the development line
    before Q20 and in the registry of 0.15.0), the value ``concrete_shell``
    reports for that length."""
    from groundfield.generators.materialise import concrete_shell
    from groundfield.geometry.electrodes import shell_resistance_from_coefficient

    spec = _foundation("lumped", 1000.0, style="mesh", n_x=2, n_y=2)
    shells = [
        shell_resistance_from_coefficient(e)
        for e in _mesh_wires(spec)
        if e.concrete_shell_coefficient_ohm_m > 0.0
    ]
    in_parallel = 1.0 / sum(1.0 / r for r in shells)
    assert in_parallel == pytest.approx(_coefficient(1000.0) / 60.0, rel=1e-12)
    assert concrete_shell(spec, encased_length_m=60.0) == pytest.approx(
        (R_A + T, _coefficient(1000.0) / 60.0, _coefficient(1000.0)), rel=1e-15
    )


# audit D2B-09
@pytest.mark.finding("10.13")
@pytest.mark.parametrize(
    ("style", "n_x", "n_y", "encased_length_m"),
    [
        ("mesh", 2, 2, 48.0),
        ("mesh", 3, 1, 44.0),
        ("mesh", 1, 1, 32.0),
        ("ring", 3, 2, 32.0),
    ],
    ids=["mesh-2x2", "mesh-3x1", "mesh-1x1", "ring"],
)
def test_private_shell_helper_spans_every_encased_wire(
    style: str, n_x: int, n_y: int, encased_length_m: float
) -> None:
    """``grounding._maybe_concrete_shell``, kept for callers of the
    pre-2026.11 private helper, still reported the shell over the perimeter
    only after Q20 (C / 32 m instead of C / 44 m for the 10 m x 6 m 3 x 1
    mesh). It reports what ``concrete_shell`` gives for the encased length
    of the materialiser, (n_y + 1) d_x + (n_x + 1) d_y, which is the per-wire
    shells of the built foundation in parallel."""
    from groundfield.generators.grounding import _maybe_concrete_shell
    from groundfield.generators.materialise import concrete_shell
    from groundfield.geometry.electrodes import shell_resistance_from_coefficient

    spec = _foundation(
        "lumped", 1000.0, style=style, size_xy_m=(10.0, 6.0), n_x=n_x, n_y=n_y
    )
    helper = _maybe_concrete_shell(spec, dx=10.0, dy=6.0, rng=np.random.default_rng(0))
    assert helper == pytest.approx(
        concrete_shell(spec, encased_length_m=encased_length_m), rel=1e-15
    )
    assert helper[1] == pytest.approx(_coefficient(1000.0) / encased_length_m)
    built = [
        shell_resistance_from_coefficient(e)
        for e in _mesh_wires(spec)
        if e.concrete_shell_coefficient_ohm_m > 0.0
    ]
    assert sum(e for e in (1.0 / r for r in built)) == pytest.approx(
        1.0 / helper[1], rel=1e-12
    )
    # An explicit length (a building outline) is the encased length as given.
    assert _maybe_concrete_shell(
        spec, dx=10.0, dy=6.0, rng=np.random.default_rng(0), perimeter_m=42.0
    )[1] == pytest.approx(_coefficient(1000.0) / 42.0)


def _house_mesh(rho_c: float) -> FoundationElectrodeSpec:
    """The generator's default house mesh (10 m, 2 x 2, 0.8 m, 5 mm) in a shell."""
    return FoundationElectrodeSpec(
        style="mesh",
        size_m=10.0,
        depth_m=0.8,
        n_x=2,
        n_y=2,
        wire_radius_m=0.005,
        concrete_rho_ohm_m=rho_c,
        concrete_thickness_m=T,
    )


def _assembled_mesh_impedance(soil: object, rho_c: float) -> float:
    """Second method: the six wire lines of the house mesh built by hand as
    separate shell-free electrodes of radius r_b, their mutual matrix from
    ``solve_mutual_matrix`` (galerkin), the shell added as ``diag(C / L_j)``
    and the wires bonded: ``1 / (1^T Z^-1 1)``."""
    r_a, depth, xs = 0.005, 0.8, (-5.0, 0.0, 5.0)
    lines = [[(x, y) for x in xs] for y in xs] + [[(x, y) for y in xs] for x in xs]
    world = gf.create_world(soil=soil)
    names = []
    for k, local in enumerate(lines):
        names.append(f"w{k}")
        gf.create_electrode(
            world,
            "polyline",
            name=names[-1],
            vertices=[(x, y, depth) for x, y in local],
            closed=False,
            wire_radius=r_a + T,
        )
    z = np.asarray(
        gf.solve_mutual_matrix(world, gf.create_engine(frequencies=[50.0]), names)
    )
    z = z + np.eye(len(names)) * _coefficient(rho_c, r_a=r_a) / 10.0  # L_j = 10 m
    ones = np.ones(len(names))
    return float((1.0 / (ones @ np.linalg.solve(z, ones))).real)


# user decision Q20
@pytest.mark.finding("10.13")
@pytest.mark.parametrize(
    "soil",
    [gf.HomogeneousSoil(rho=100.0), gf.TwoLayerSoil(rho_1=30.0, rho_2=100.0, h_1=5.0)],
    ids=["homogeneous", "two-layer"],
)
@pytest.mark.parametrize("rho_c", [150.0, 5000.0])
def test_encased_mesh_equals_the_assembly_of_its_wires(
    soil: object, rho_c: float
) -> None:
    """``Engine.solve`` on the generated default house mesh and the hand
    assembly of its six encased wires agree to round-off (a check of the
    generator's wiring: the same kernel and owner model, audit D2B-07).
    With the braces bare (the development line up to Q20) the solve was
    -1.4 / -75 % (100 ohm m) and -11 / -90 % (30 / 100 ohm m, h_1 = 5 m)
    below the assembly at rho_c = 150 / 5000 ohm m (evidence d2gen
    ``brace_effect_base.out``)."""
    world = gf.create_world(soil=soil)
    anchor = GroundingSystemSpec(electrodes=[_house_mesh(rho_c)]).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    gf.create_source(world, attached_to=anchor, magnitude=1.0)
    z_solve = gf.create_engine(frequencies=[50.0]).solve(world).input_impedance()[0]
    assert z_solve.real == pytest.approx(
        _assembled_mesh_impedance(soil, rho_c), rel=1e-12
    )


#: Input impedance (ohm) of the generator default house mesh (10 m, 2 x 2,
#: 0.8 m, r_a = 5 mm, t = 5 cm) built with ``build_at``, 1 A at its anchor,
#: solved by groundfield 0.15.0 with its default engine (image family, point
#: kernel, 0.5 m segments, 50 Hz), at rho_c = 50 / 100 / 150 / 1000 / 5000
#: ohm m. Evidence f2gen ``mesh_vs_v0150.py`` on the v0.15.0 checkout,
#: reproduced by the auditor's script (D2B s01) and, to round-off, by the
#: 0.15.0 models rebuilt in this release (``mesh_vs_v0150_models.out``).
V0150_HOUSE_MESH = {
    ("100 Ω·m", "lumped"): (4.072060, 4.072060, 4.072060, 4.072060, 4.072060),
    ("100 Ω·m", "distributed"): (4.260941, 4.495502, 4.699200, 6.226285, 7.199813),
    ("30 / 100 Ω·m", "lumped"): (1.870050, 1.870050, 1.870050, 1.870050, 1.870050),
    ("30 / 100 Ω·m", "distributed"): (2.079758, 2.238138, 2.349233, 2.803409, 2.941482),
}
HOUSE_RHO_C = (50.0, 100.0, 150.0, 1000.0, 5000.0)
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _percentages(row: str) -> list[tuple[float, float]]:
    """The signed percentages of a Markdown table row ("+1 701 %" too), each
    with half a unit of its last printed digit."""
    out = []
    for text in re.findall(r"([+−-][0-9][0-9 .]*) %", row):
        text = text.replace(" ", "").replace("−", "-")
        decimals = len(text.split(".")[1]) if "." in text else 0
        out.append((float(text), 0.5 * 10.0**-decimals))
    return out


# audit D2B-01
@pytest.mark.finding("10.13")
def test_texts_place_the_bare_braces_on_the_development_line() -> None:
    """In 0.15.0 the braces of a generated mesh foundation were never bare:
    every wire had r_b, "distributed" put C on the perimeter only and
    "lumped" on no wire. The bare braces (r_a, no C) existed only on the
    2026.11 development line before user decision Q20, but the field
    description, the module docstring, the concrete-shell page and ADR-0012
    said "up to then the braces stayed bare"."""
    from groundfield.generators import materialise

    texts = {
        "field description": FoundationElectrodeSpec.model_fields[
            "concrete_rho_ohm_m"
        ].description,
        "materialise module": materialise.__doc__,
        "concrete_shell": materialise.concrete_shell.__doc__,
        "concrete-shell page": (DOCS / "explanation" / "concrete-shells.md").read_text(
            encoding="utf-8"
        ),
        "ADR-0012": (DOCS / "adr" / "0012-foundation-concrete-encasement.md").read_text(
            encoding="utf-8"
        ),
    }
    for name, text in texts.items():
        flat = " ".join(text.split())
        assert "development line" in flat, name
        assert "0.15.0" in flat, name
        for stale in (
            "up to then the braces stayed bare",
            "the earlier rule kept the braces bare",
            "up to then the braces kept the bare radius",
        ):
            assert stale not in flat, (name, stale)


# audit D2B-01
@pytest.mark.finding("10.13")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize(
    ("soil_name", "soil"),
    [
        ("100 Ω·m", gf.HomogeneousSoil(rho=100.0)),
        ("30 / 100 Ω·m", gf.TwoLayerSoil(rho_1=30.0, rho_2=100.0, h_1=5.0)),
    ],
    ids=["homogeneous", "two-layer"],
)
def test_the_concrete_shell_page_states_the_change_against_0150(
    soil_name: str, soil: object, model: str
) -> None:
    """The brace-effect table is relative to the bare braces of the
    development line; *Numbers that change* and the users of 0.15.0 need the
    change against 0.15.0, which the page states per ``concrete_model``
    value of 0.15.0 (house mesh, default engine of each version). Measured:
    100 ohm m +7.0 ... +782 % ("lumped") and +2.2 ... +399 %
    ("distributed")."""
    page = (DOCS / "explanation" / "concrete-shells.md").read_text(encoding="utf-8")
    section = page.split("### Against 0.15.0", 1)[1].split("\n## ", 1)[0]
    row = next(
        line
        for line in section.splitlines()
        if line.startswith(f"| {soil_name}") and f'`"{model}"`' in line
    )
    quoted = _percentages(row)
    assert len(quoted) == len(HOUSE_RHO_C), row
    engine = gf.create_engine(frequencies=[50.0])
    for rho_c, old, (documented, half_unit) in zip(
        HOUSE_RHO_C, V0150_HOUSE_MESH[(soil_name, model)], quoted, strict=True
    ):
        world = gf.create_world(soil=soil)
        anchor = GroundingSystemSpec(electrodes=[_house_mesh(rho_c)]).build_at(
            world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
        )
        gf.create_source(world, attached_to=anchor, magnitude=1.0)
        new = engine.solve(world).input_impedance()[0].real
        change = 100.0 * (new / old - 1.0)
        assert change == pytest.approx(documented, abs=half_unit + 1e-9)


def _per_segment_brace_effect(rho_cs: tuple[float, ...]) -> list[float]:
    """Encased over bare braces - 1 of the house mesh in 100 ohm m with one
    current per segment (all segments at one potential): every 0.25 m
    segment its own electrode, mutual matrix from ``solve_mutual_matrix``,
    the shell C / ds on the diagonal of the encased segments."""
    r_a, r_b, depth, ds = 0.005, 0.005 + T, 0.8, 0.25
    xs = (-5.0, 0.0, 5.0)
    lines = [[(x, y) for x in xs] for y in xs] + [[(x, y) for y in xs] for x in xs]
    perimeter = {0, 2, 3, 5}
    engine = gf.create_engine(frequencies=[50.0], segment_length=ds)
    results = {}
    for rule in ("bare", "encased"):
        world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
        names, encased = [], []
        for j, local in enumerate(lines):
            shell = rule == "encased" or j in perimeter
            for (x0, y0), (x1, y1) in zip(local[:-1], local[1:], strict=True):
                for k in range(20):  # 5 m spans in 0.25 m segments
                    names.append(f"s{len(names)}")
                    encased.append(shell)
                    gf.create_electrode(
                        world,
                        "strip",
                        name=names[-1],
                        start=(x0 + (x1 - x0) * k / 20, y0 + (y1 - y0) * k / 20, depth),
                        end=(
                            x0 + (x1 - x0) * (k + 1) / 20,
                            y0 + (y1 - y0) * (k + 1) / 20,
                            depth,
                        ),
                        wire_radius=r_b if shell else r_a,
                    )
        p = np.asarray(gf.solve_mutual_matrix(world, engine, names)).real
        ones = np.ones(len(names))
        results[rule] = [
            1.0
            / (
                ones
                @ np.linalg.solve(
                    p + np.diag(np.where(encased, _coefficient(rc, r_a=r_a) / ds, 0.0)),
                    ones,
                )
            )
            for rc in rho_cs
        ]
    return [
        100.0 * (e / b - 1.0)
        for e, b in zip(results["encased"], results["bare"], strict=True)
    ]


# audit D2B-07
@pytest.mark.finding("10.13")
def test_the_brace_effect_names_its_model_and_second_method() -> None:
    """The hand assembly from ``solve_mutual_matrix`` shares the kernel and
    the one-owner-per-wire model with ``Engine.solve``: it checks the
    generator, it is not an independent method. With one current per
    segment the brace effect is +49.2 / +318.5 % instead of +46.9 / +307 %
    at rho_c = 1000 / 5000 ohm m (house, 100 ohm m; the auditor's
    independent BEM agrees to 0.02 pp), a model uncertainty the page and
    ADR-0012 now state."""
    page = " ".join(
        (DOCS / "explanation" / "concrete-shells.md")
        .read_text(encoding="utf-8")
        .split()
    )
    adr = " ".join(
        (DOCS / "adr" / "0012-foundation-concrete-encasement.md")
        .read_text(encoding="utf-8")
        .split()
    )
    for text in (page, adr):
        assert "independent assembly" not in text
        assert "not the numbers" in text
    effect = _per_segment_brace_effect((1000.0, 5000.0))
    found = re.search(
        r"changes \$Z\$ by \+([0-9.]+) / \+([0-9.]+) % instead of \+46\.9 / \+307 %",
        page,
    )
    assert found, "per-segment brace effect not quoted"
    assert [float(found.group(1)), float(found.group(2))] == pytest.approx(
        effect, abs=0.05
    )
    assert "+49.2 / +318.5 % instead of +46.9 / +307 %" in adr


@pytest.mark.finding("10.13")
def test_mesh_foundation_lumped_equals_distributed() -> None:
    """v0.15.0 default mesh foundation: lumped vs distributed +9.6 / +56 /
    +148 % (braces through the shell, inflated brace radius)."""
    for rho_c in (150.0, 1000.0, 5000.0):
        z = {
            m: _solve_site([_foundation(m, rho_c, style="mesh", n_x=2, n_y=2)])
            for m in ("lumped", "distributed")
        }
        assert z["lumped"] == pytest.approx(z["distributed"], rel=1e-12)


@pytest.mark.finding("10.13")
def test_substation_and_cabinet_shells_are_consumed() -> None:
    """Encased substation / cabinet foundations: v0.15.0 kept their shell in
    the registry, which no PEN drop read (rho_c 150 vs 5000 bit-identical)."""

    def z_in(rho_c: float) -> complex:
        encased = GroundingSystemSpec(
            electrodes=[
                _foundation("lumped", rho_c, size_m=6.0),
                RodElectrodeSpec(length_m=2.5, offset_xy_m=(2.0, 0.0)),
            ]
        )
        cfg = TnNetworkConfig(
            soil=HomogeneousSoilSpec(rho=100.0),
            substation=SubstationConfig(grounding=encased),
            cabinets=CabinetConfig(
                fixed_count=1,
                placement=ExplicitPlacement(positions=[(40.0, 0.0)]),
                grounding=encased,
            ),
            building_counts={"residential": 2},
        )
        world = TnNetworkGenerator(cfg, seed=0).build()
        assert world.concrete_shell_corrections == {}
        return _engine().solve(world).input_impedance()[0]

    assert z_in(5000.0).real > 1.05 * z_in(150.0).real


@pytest.mark.finding("10.12")
def test_service_drops_carry_no_shell_resistance() -> None:
    """The shell sits on the foundation; the PEN drop keeps its geometric
    resistance (v0.15.0 put R_shell on it)."""
    building = BuildingTypeSpec(
        name="house",
        grounding=GroundingSystemSpec(electrodes=[_foundation("lumped", 5000.0)]),
    )
    cfg = TnNetworkConfig(building_types=[building], building_counts={"house": 2})
    world = TnNetworkGenerator(cfg, seed=0).build()
    # The ring foundations' anchors lie on the PEN main, which runs through
    # them (no drops since the PEN routing of 2026.11): check every section.
    pen = [c for c in world.conductors if c.conductor_type == "pen"]
    assert pen and all(c.lumped_series_resistance_ohm is None for c in pen)


# ---------------------------------------------------------------------
# 10.14: the layout builder honours the shell
# ---------------------------------------------------------------------


@pytest.mark.finding("10.14")
def test_layout_builder_honours_the_shell() -> None:
    """v0.15.0 (F08): rho_c = 150 and 5000 gave bit-identical layout worlds'
    results."""
    footprints = [
        gf.BuildingFootprint(
            polygon_xy_m=[
                (x - 5.0, 13.0),
                (x + 5.0, 13.0),
                (x + 5.0, 23.0),
                (x - 5.0, 23.0),
            ],
            osm_id=i,
        )
        for i, x in enumerate((20.0, 45.0))
    ]
    layout = LvNetworkLayout.from_footprints(footprints, substation_xy=(0.0, 0.0))
    layout.add_pen_cable(start="substation", end_xy=(60.0, 0.0))
    layout.connect_buildings()

    def z_in(rho_c: float) -> complex:
        house = GroundingSystemSpec(electrodes=[_foundation("lumped", rho_c)])
        world = layout.to_world(seed=0, house_grounding=house)
        assert world.concrete_shell_corrections == {}
        foundation = world.get_electrode("house_000_foundation_0")
        assert foundation.concrete_shell_coefficient_ohm_m == pytest.approx(
            _coefficient(rho_c)
        )
        return _engine().solve(world).input_impedance()[0]

    assert z_in(5000.0).real > 1.05 * z_in(150.0).real
