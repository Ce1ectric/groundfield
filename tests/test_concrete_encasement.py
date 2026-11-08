"""Tests for ADR-0012: concrete encasement of foundation electrodes.

Covers the per-segment shell coefficient (diagonal augmentation in the
image / image_2layer backends) that both ``concrete_model`` values use
since 2026.11 — up to 0.15.0 V1 (``"lumped"``) was a series resistance on
the PEN service drop instead (review findings 10.12–10.14) — plus the
closed-form Sunde-shell formula and the end-to-end stochastic moisture
path.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.generators import (
    BuildingTypeSpec,
    Discrete,
    ExplicitPlacement,
    FoundationElectrodeSpec,
    GroundingSystemSpec,
    KvsConfig,
    SubstationConfig,
    TnNetworkConfig,
    TnNetworkGenerator,
    TwoLayerSoilSpec,
)
from groundfield.geo import BuildingFootprint, OsmBuildingPlacement
from groundfield.geometry.electrodes import StripElectrode


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _solve_z(world, *, segment_length=0.5) -> float:
    """Solve at 50 Hz, return the source-cluster impedance in Ω."""
    eng = gf.create_engine(
        backend="image_2layer",
        segment_length=segment_length,
        frequencies=[50.0],
        earth_inductive_model="perfect_mirror",
    )
    result = eng.solve(world)
    cluster = world.electrodes[0].name
    return result.cluster_impedance(cluster)[0].real


# ---------------------------------------------------------------------
# V2 — closed-form Sunde shell on an isolated strip
# ---------------------------------------------------------------------


def test_v2_distributed_matches_sunde_closed_form_on_isolated_strip() -> None:
    """An isolated 10 m horizontal strip with shell coefficient $C$
    must show ``Z_with_shell - Z_without_shell == C / L`` to within
    numerical noise. Independent of the soil resistivity (the
    augmentation enters the MoM diagonal post-kernel and does not
    couple to the bulk kernel)."""
    L = 10.0
    for C in (55.0, 500.0, 5000.0):
        w_bare = gf.create_world(
            name="bare",
            soil=gf.TwoLayerSoil(rho_1=150.0, rho_2=30.0, h_1=4.0),
        )
        w_bare.add_electrode(
            StripElectrode(
                name="s",
                start=(-L / 2, 0.0, 0.8),
                end=(L / 2, 0.0, 0.8),
                wire_radius=0.055,
            )
        )
        gf.create_source(w_bare, attached_to="s", magnitude=1.0)
        z_bare = _solve_z(w_bare)

        w_shell = gf.create_world(
            name="shell",
            soil=gf.TwoLayerSoil(rho_1=150.0, rho_2=30.0, h_1=4.0),
        )
        w_shell.add_electrode(
            StripElectrode(
                name="s",
                start=(-L / 2, 0.0, 0.8),
                end=(L / 2, 0.0, 0.8),
                wire_radius=0.055,
                concrete_shell_coefficient_ohm_m=C,
            )
        )
        gf.create_source(w_shell, attached_to="s", magnitude=1.0)
        z_shell = _solve_z(w_shell)

        expected_addition = C / L
        assert z_shell - z_bare == pytest.approx(
            expected_addition, rel=1e-4, abs=1e-3
        ), f"C={C}: expected +{expected_addition} Ω, got +{z_shell - z_bare}"


# ---------------------------------------------------------------------
# V2 — vanishing and insulating limits
# ---------------------------------------------------------------------


def test_v2_zero_coefficient_is_no_op() -> None:
    """Setting ``concrete_shell_coefficient_ohm_m = 0`` must leave the
    result bit-exact equal to the no-shell case — the
    ``np.any(... > 0)`` short-circuit in the solver skips the
    augmentation entirely."""
    L = 10.0
    w0 = gf.create_world(
        name="w0", soil=gf.TwoLayerSoil(rho_1=150.0, rho_2=30.0, h_1=4.0)
    )
    w0.add_electrode(
        StripElectrode(
            name="s",
            start=(-L / 2, 0, 0.8),
            end=(L / 2, 0, 0.8),
            wire_radius=0.055,
        )
    )
    gf.create_source(w0, attached_to="s", magnitude=1.0)

    w_zero = gf.create_world(
        name="w_zero", soil=gf.TwoLayerSoil(rho_1=150.0, rho_2=30.0, h_1=4.0)
    )
    w_zero.add_electrode(
        StripElectrode(
            name="s",
            start=(-L / 2, 0, 0.8),
            end=(L / 2, 0, 0.8),
            wire_radius=0.055,
            concrete_shell_coefficient_ohm_m=0.0,
        )
    )
    gf.create_source(w_zero, attached_to="s", magnitude=1.0)

    assert _solve_z(w0) == pytest.approx(_solve_z(w_zero), abs=1e-12)


def test_v2_insulating_concrete_dominates_impedance() -> None:
    """For a very large shell coefficient (dry concrete, $\\rho_c =
    10\\,000\\,\\Omega\\cdot\\text{m}$ × ln(11)/(2π)), the foundation's
    cluster impedance is dominated by the lumped shell term and
    follows ``Z ≈ C / L`` to within a few percent (the bulk-soil
    contribution becomes negligible)."""
    L = 10.0
    C = 5000.0
    w = gf.create_world(
        name="w", soil=gf.TwoLayerSoil(rho_1=150.0, rho_2=30.0, h_1=4.0)
    )
    w.add_electrode(
        StripElectrode(
            name="s",
            start=(-L / 2, 0, 0.8),
            end=(L / 2, 0, 0.8),
            wire_radius=0.055,
            concrete_shell_coefficient_ohm_m=C,
        )
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    z = _solve_z(w)
    # C / L = 500 Ω; bulk-soil contribution roughly 7-12 Ω.
    assert z > C / L
    assert z < C / L + 30.0


# ---------------------------------------------------------------------
# V1 — the lumped shell is the Sunde coefficient on the foundation
# ---------------------------------------------------------------------


def _one_house_cfg(name: str, foundation: FoundationElectrodeSpec) -> TnNetworkConfig:
    residential = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(electrodes=[foundation]),
    )
    return TnNetworkConfig(
        name=name,
        soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
        substation=SubstationConfig(position=(40.0, -25.0)),
        placement=ExplicitPlacement(positions=[(0.0, 10.0)]),
        building_types=[residential],
        building_counts={"residential": 1},
        kvs=KvsConfig(
            fixed_count=1,
            placement=ExplicitPlacement(positions=[(40.0, -8.0)]),
        ),
    )


@pytest.mark.finding("10.12")
def test_v1_lumped_shell_is_the_sunde_coefficient_on_the_foundation() -> None:
    r"""``concrete_model='lumped'`` puts the Sunde coefficient
    $C = \rho_c/(2\pi)\,\ln(r_b/r_a)$ on the foundation (radius $r_b$);
    its lumped resistance $C/L_\text{perim}$ matches the closed form. Up to
    0.15.0 the value went into ``World.concrete_shell_corrections``
    (10.12). Since user decision Q38 the ring is one strip per side, each
    with $r_b$ and $C$; their shells $C/L_k$ in parallel are
    $C/L_\text{perim}$ (up to then one grid mesh carried it)."""
    from groundfield.geometry.electrodes import shell_resistance_from_coefficient

    rho_c = 5000.0
    t = 0.05
    dx_dy = (10.0, 10.0)
    r_a, r_b = 0.005, 0.005 + t
    perimeter = 2.0 * (dx_dy[0] + dx_dy[1])
    expected_r_shell = rho_c / (2.0 * math.pi * perimeter) * math.log(r_b / r_a)

    world = TnNetworkGenerator(
        _one_house_cfg(
            "t1",
            FoundationElectrodeSpec(
                style="ring",
                size_xy_m=dx_dy,
                depth_m=0.8,
                concrete_rho_ohm_m=rho_c,
                concrete_thickness_m=t,
                concrete_model="lumped",
            ),
        ),
        seed=0,
    ).build()
    assert world.concrete_shell_corrections == {}
    name = "residential_0_foundation_0"
    sides = [world.get_electrode(name)] + [
        world.get_electrode(f"{name}_w{j}") for j in (1, 2, 3)
    ]
    assert [side.kind for side in sides] == ["strip"] * 4
    for side in sides:
        assert side.wire_radius == pytest.approx(r_b)
        assert side.concrete_shell_coefficient_ohm_m == pytest.approx(
            rho_c / (2.0 * math.pi) * math.log(r_b / r_a), rel=1e-15
        )
    in_parallel = 1.0 / sum(
        1.0 / shell_resistance_from_coefficient(side) for side in sides
    )
    assert in_parallel == pytest.approx(expected_r_shell, rel=1e-12)


@pytest.mark.finding("10.12")
def test_v1_pen_service_drop_keeps_its_geometric_resistance() -> None:
    """The shell sits on the foundation, so the PEN service drop carries
    no ``lumped_series_resistance_ohm`` any more (it did up to 0.15.0,
    which also sent the current of bonded rods through the shell, 10.13)."""
    world = TnNetworkGenerator(
        _one_house_cfg(
            "t2",
            FoundationElectrodeSpec(
                style="ring",
                size_xy_m=(10.0, 10.0),
                depth_m=0.8,
                concrete_rho_ohm_m=5000.0,
                concrete_thickness_m=0.05,
                concrete_model="lumped",
            ),
        ),
        seed=0,
    ).build()
    bond = next(
        c
        for c in world.conductors
        if c.name == "pen_service_residential_0_foundation_0"
    )
    assert bond.lumped_series_resistance_ohm is None
    assert bond.series_resistance == pytest.approx(
        1.68e-8 * bond.length / (math.pi * bond.wire_radius**2)
    )


# ---------------------------------------------------------------------
# Model discriminator: lumped vs. distributed bookkeeping
# ---------------------------------------------------------------------


def test_distributed_model_does_not_use_lumped_registry() -> None:
    """``concrete_model='distributed'`` writes to the strips'
    ``concrete_shell_coefficient_ohm_m`` field — one owner per wire —
    and never to the ``world.concrete_shell_corrections`` registry (which
    no generator fills since 2026.11)."""
    residential = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(
                    style="ring",
                    size_xy_m=(10.0, 10.0),
                    depth_m=0.8,
                    concrete_rho_ohm_m=5000.0,
                    concrete_thickness_m=0.05,
                    concrete_model="distributed",
                ),
            ],
        ),
    )
    cfg = TnNetworkConfig(
        name="t3",
        soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
        substation=SubstationConfig(position=(40.0, -25.0)),
        placement=ExplicitPlacement(positions=[(0.0, 10.0)]),
        building_types=[residential],
        building_counts={"residential": 1},
        kvs=KvsConfig(
            fixed_count=1,
            placement=ExplicitPlacement(positions=[(40.0, -8.0)]),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    # Registry empty for the distributed path
    assert world.concrete_shell_corrections == {}
    # Strips carry the per-meter Sunde coefficient
    strips = [
        e for e in world.electrodes if e.kind == "strip" and "foundation" in e.name
    ]
    assert len(strips) == 4
    expected_coeff = 5000.0 / (2.0 * math.pi) * math.log(0.055 / 0.005)
    for s in strips:
        assert s.concrete_shell_coefficient_ohm_m == pytest.approx(
            expected_coeff, rel=1e-6
        )


# ---------------------------------------------------------------------
# JSON round-trip — including stochastic moisture
# ---------------------------------------------------------------------


def test_concrete_fields_json_roundtrip() -> None:
    """A spec with stochastic ``concrete_rho_ohm_m`` must round-trip
    through Pydantic's JSON adapter."""
    spec = FoundationElectrodeSpec(
        style="ring",
        size_xy_m=(10.0, 10.0),
        depth_m=0.8,
        concrete_rho_ohm_m=Discrete(
            values=[50.0, 150.0, 500.0, 2000.0],
            weights=[0.25, 0.40, 0.25, 0.10],
        ),
        concrete_thickness_m=0.05,
        concrete_model="distributed",
    )
    payload = spec.model_dump_json()
    restored = FoundationElectrodeSpec.model_validate_json(payload)
    assert restored.concrete_model == "distributed"
    assert restored.concrete_thickness_m == 0.05
    assert isinstance(restored.concrete_rho_ohm_m, Discrete)
    assert list(restored.concrete_rho_ohm_m.values) == [50.0, 150.0, 500.0, 2000.0]


def test_stochastic_moisture_samples_per_realisation() -> None:
    """Building two worlds with the same seed and the same
    ``Discrete`` distribution on ``concrete_rho_ohm_m`` must produce
    identical electrode counts and identical shell coefficients.
    Building with a different seed must produce different sampled
    values."""
    discrete = Discrete(values=[50.0, 5000.0], weights=[0.5, 0.5])
    residential = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(
                    style="ring",
                    size_xy_m=(10.0, 10.0),
                    depth_m=0.8,
                    concrete_rho_ohm_m=discrete,
                    concrete_thickness_m=0.05,
                    concrete_model="lumped",
                ),
            ],
        ),
    )
    cfg = TnNetworkConfig(
        name="t4",
        soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
        substation=SubstationConfig(position=(40.0, -25.0)),
        placement=ExplicitPlacement(
            positions=[(0.0, 10.0), (20.0, 10.0), (40.0, 10.0)],
        ),
        building_types=[residential],
        building_counts={"residential": 3},
        kvs=KvsConfig(
            fixed_count=1,
            placement=ExplicitPlacement(positions=[(40.0, -8.0)]),
        ),
    )

    def shells(world):
        return {
            e.name: e.concrete_shell_coefficient_ohm_m
            for e in world.electrodes
            if "foundation" in e.name
        }

    w_a = TnNetworkGenerator(cfg, seed=0).build()
    w_b = TnNetworkGenerator(cfg, seed=0).build()
    assert shells(w_a) == shells(w_b)
    # Three foundations of one strip per side each (user decision Q38; up to
    # then one grid mesh each); the four sides share one draw.
    assert len(shells(w_a)) == 12
    for k in range(3):
        name = f"residential_{k}_foundation_0"
        assert len({shells(w_a)[n] for n in shells(w_a) if n.startswith(name)}) == 1

    w_c = TnNetworkGenerator(cfg, seed=1).build()
    # At least one foundation should have a different sampled shell
    # (the moisture draw is per-foundation).
    diffs = {k: shells(w_a)[k] - shells(w_c)[k] for k in shells(w_a)}
    assert any(abs(d) > 1e-6 for d in diffs.values())


# ---------------------------------------------------------------------
# Non-foundation electrodes never get the shell
# ---------------------------------------------------------------------


def test_substation_ring_and_kvs_rod_are_never_concrete_encased() -> None:
    """Only ``FoundationElectrodeSpec`` carries the concrete fields;
    the substation ring + rods and the KVS rod get no shell coefficient
    (and no generator writes ``world.concrete_shell_corrections``)."""
    residential = BuildingTypeSpec(
        name="residential",
        grounding=GroundingSystemSpec(
            electrodes=[
                FoundationElectrodeSpec(
                    style="ring",
                    size_xy_m=(10.0, 10.0),
                    depth_m=0.8,
                    concrete_rho_ohm_m=5000.0,
                    concrete_thickness_m=0.05,
                    concrete_model="lumped",
                ),
            ],
        ),
    )
    cfg = TnNetworkConfig(
        name="t5",
        soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
        substation=SubstationConfig(position=(40.0, -25.0)),
        placement=ExplicitPlacement(positions=[(0.0, 10.0)]),
        building_types=[residential],
        building_counts={"residential": 1},
        kvs=KvsConfig(
            fixed_count=1,
            placement=ExplicitPlacement(positions=[(40.0, -8.0)]),
        ),
    )
    world = TnNetworkGenerator(cfg, seed=0).build()
    assert world.concrete_shell_corrections == {}
    # The only shell is on the foundation (one strip per side since user
    # decision Q38), not on any substation / KVS electrode.
    encased = [e.name for e in world.electrodes if e.concrete_shell_coefficient_ohm_m]
    assert encased == ["residential_0_foundation_0"] + [
        f"residential_0_foundation_0_w{j}" for j in (1, 2, 3)
    ]


# ---------------------------------------------------------------------
# End-to-end: OSM-driven world with concrete shell increases impedance
# ---------------------------------------------------------------------


def _rotated_rectangle(centre, size, angle_deg):
    cx, cy = centre
    dx, dy = size
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    local = [(-dx / 2, -dy / 2), (dx / 2, -dy / 2), (dx / 2, dy / 2), (-dx / 2, dy / 2)]
    return [(c * x - s * y + cx, s * x + c * y + cy) for x, y in local]


def test_osm_pipeline_with_dry_concrete_drastically_increases_z_system() -> None:
    """End-to-end smoke test: in an OSM-driven Ortsnetz, dry concrete
    (ρ_c = 5000 Ω·m) decouples the foundations from the substation
    grounding system. The substation's effective system impedance —
    measured as ``phi_sub / I_source`` — therefore rises sharply
    (factor ~3–5×) compared with the no-concrete case."""
    buildings = [
        BuildingFootprint(polygon_xy_m=_rotated_rectangle(*p))
        for p in [
            ((0, 10), (12, 8), -15.0),
            ((20, 11), (10, 9), 10.0),
            ((40, 12), (11, 8), -5.0),
            ((60, 11), (13, 7), 15.0),
            ((80, 10), (10, 10), 25.0),
        ]
    ]
    common_cfg_kwargs = dict(
        name="osm_concrete",
        soil=TwoLayerSoilSpec(rho_1=150.0, rho_2=30.0, h_1=4.0),
        substation=SubstationConfig(position=(40.0, -25.0)),
        placement=OsmBuildingPlacement(footprints=buildings, min_area_m2=16.0),
        building_counts={"residential": 5},
        kvs=KvsConfig(
            fixed_count=1,
            placement=ExplicitPlacement(positions=[(40.0, -8.0)]),
        ),
    )

    def _phi_sub(rho_c):
        residential = BuildingTypeSpec(
            name="residential",
            grounding=GroundingSystemSpec(
                electrodes=[
                    FoundationElectrodeSpec(
                        style="ring",
                        size_m=10.0,
                        depth_m=0.8,
                        concrete_rho_ohm_m=rho_c,
                        concrete_thickness_m=0.05,
                        concrete_model="lumped",
                    ),
                ],
            ),
        )
        cfg = TnNetworkConfig(building_types=[residential], **common_cfg_kwargs)
        world = TnNetworkGenerator(cfg, seed=0).build()
        eng = gf.create_engine(
            backend="image_2layer",
            segment_length=0.5,
            frequencies=[50.0],
            earth_inductive_model="perfect_mirror",
        )
        return eng.solve(world).electrode_potentials["trafo_ring_0"][0].real

    phi_no_concrete = _phi_sub(None)
    phi_dry = _phi_sub(5000.0)

    # Dry concrete must give a *substantially* higher system
    # impedance than the no-concrete reference.
    assert phi_dry > 2.5 * phi_no_concrete, (
        f"expected phi_dry / phi_no_concrete > 2.5; "
        f"got {phi_no_concrete:.3f} vs {phi_dry:.3f}"
    )
