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
Since 2026.11 both models put :math:`C` on the encased wires themselves;
cross-braces of a mesh foundation are not encased and keep the bare radius.
"""

from __future__ import annotations

import math
import warnings

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
    probes = np.array([(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)  # no registry entries
        z = gf.solve_mutual_matrix(world, engine, anchors, probes)
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


@pytest.mark.finding("10.13")
@pytest.mark.parametrize("model", ["lumped", "distributed"])
@pytest.mark.parametrize("orientation_deg", [None, 30.0])
def test_mesh_braces_are_not_encased(model: str, orientation_deg: float | None) -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    spec = _foundation(model, 500.0, style="mesh", n_x=2, n_y=2)
    spec = spec.model_copy(update={"orientation_deg": orientation_deg})
    GroundingSystemSpec(electrodes=[spec]).build_at(
        world, site_xy=(0.0, 0.0), name_prefix="h", rng=np.random.default_rng(0)
    )
    wires = [e for e in world.electrodes if e.name.startswith("h_foundation")]
    assert len(wires) == 6  # 4 perimeter wires + 2 braces
    encased = [e for e in wires if e.concrete_shell_coefficient_ohm_m > 0.0]
    braces = [e for e in wires if e.concrete_shell_coefficient_ohm_m == 0.0]
    assert len(encased) == 4 and len(braces) == 2
    assert all(e.wire_radius == pytest.approx(R_A + T) for e in encased)
    assert all(e.wire_radius == pytest.approx(R_A) for e in braces)


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
    drops = [c for c in world.conductors if c.name.startswith("pen_service_")]
    assert drops and all(c.lumped_series_resistance_ohm is None for c in drops)


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
