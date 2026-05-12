"""End-to-end smoke test of the top-level API + image backend.

Mirrors ``notebooks/01_smoke_test.ipynb`` and breaks the CI as soon as
the user-facing API regresses or the image solution drifts too far
from the analytical Sunde formula.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import groundfield as gf


# ---------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------


def test_top_level_exports() -> None:
    needed = {
        "create_world",
        "create_electrode",
        "create_conductor",
        "create_source",
        "create_engine",
        "run_simulation",
        "plot_potential_contour",
        "plot_potential_profile",
        "plot_potential_radial",
        "World",
        "Engine",
        "FieldResult",
        "HomogeneousSoil",
        "TwoLayerSoil",
        "RodElectrode",
        "RingElectrode",
        "Conductor",
        "CurrentSource",
        "BoundaryConditions",
    }
    assert needed.issubset(set(gf.__all__))


def test_full_notebook_workflow() -> None:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(name="ap1_minimal", soil=soil)

    g1 = gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.5
    )
    g2 = gf.create_electrode(
        world, "ring", name="g2", center=(10.0, 0.0, 0.8), radius=2.0
    )
    assert world.get_electrode("g1") is g1
    assert world.get_electrode("g2") is g2

    l1 = gf.create_conductor(
        world, name="l1", start=g1, end=g2, conductor_type="bare_copper"
    )
    assert l1.length == pytest.approx(math.dist((0, 0, 0.5), (12, 0, 0.8)))

    s1 = gf.create_source(world, name="s1", attached_to=g1, magnitude=10.0)
    assert s1.attached_to == "g1"

    world.set_boundary_conditions(far_field="dirichlet")
    eng = gf.create_engine(backend="image", segment_length=0.05)
    result = gf.run_simulation(world, eng)

    assert result.backend == "image"
    assert set(result.electrode_potentials.keys()) == {"g1", "g2"}
    # With the conductor present, g1 and g2 share a cluster — current
    # is split between the two electrodes.
    I_total = result.electrode_currents["g1"][0] + result.electrode_currents["g2"][0]
    assert I_total.real == pytest.approx(10.0)
    assert sorted(result.clusters["g1"]) == ["g1", "g2"]
    assert abs(result.electrode_potentials["g1"][0]
               - result.electrode_potentials["g2"][0]) < 1e-9
    # The result must contain the discretised point sources.
    assert len(result.point_sources) > 0


def test_world_solve_method_equivalent() -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=50.0))
    gf.create_electrode(world, "rod", name="g1", position=(0.0, 0.0, 0.5), length=1.0)
    gf.create_source(world, name="s1", attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(segment_length=0.05)
    r1 = world.solve(eng)
    r2 = gf.run_simulation(world, eng)
    assert r1.electrode_potentials == r2.electrode_potentials


def test_auto_naming() -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    e0 = gf.create_electrode(world, "rod", position=(0.0, 0.0, 0.5), length=1.0)
    e1 = gf.create_electrode(world, "rod", position=(1.0, 0.0, 0.5), length=1.0)
    assert e0.name == "electrode_0"
    assert e1.name == "electrode_1"


def test_solve_without_soil_raises() -> None:
    world = gf.create_world()
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    with pytest.raises(ValueError, match="soil"):
        world.solve(gf.create_engine())


def test_solve_without_electrodes_raises() -> None:
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    with pytest.raises(ValueError, match="electrodes"):
        world.solve(gf.create_engine())


def test_two_layer_soil_reflection_coefficient() -> None:
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=2.0)
    expected = (1000.0 - 100.0) / (1000.0 + 100.0)
    assert soil.reflection_coefficient == pytest.approx(expected)


# ---------------------------------------------------------------------
# Image backend: plausibility
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "L, wire_radius",
    [(1.5, 0.005), (3.0, 0.005), (1.5, 0.01)],
)
def test_image_rod_matches_sunde(L: float, wire_radius: float) -> None:
    """The image backend must agree with Sunde to within 10 %."""
    rho = 100.0
    soil = gf.HomogeneousSoil(resistivity=rho)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1",
        position=(0.0, 0.0, 0.5), length=L, wire_radius=wire_radius,
    )
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    result = eng.solve(world)

    Z = result.grounding_impedance("g1")[0].real
    d = 2.0 * wire_radius
    R_sunde = rho / (2.0 * math.pi * L) * (math.log(4.0 * L / d) - 1.0)
    rel_err = abs(Z - R_sunde) / R_sunde
    assert rel_err < 0.10, (
        f"Z = {Z:.2f} Ω deviates by > 10 % from Sunde = {R_sunde:.2f} Ω"
    )


def test_image_potential_decays_monotonically() -> None:
    """Potential must decay monotonically with distance."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)
    result = eng.solve(world)

    xs = np.linspace(1.0, 50.0, 60)
    pts = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
    phi = result.potential(pts).real
    diffs = np.diff(phi)
    assert (diffs <= 1e-9).all(), f"Potential not monotonic: max diff = {diffs.max()}"


def test_image_neighbour_induces_potential() -> None:
    """A passive neighbour picks up an induced potential > 0 and < own EPR."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_electrode(world, "ring", name="g2",
                        center=(8.0, 0, 0.8), radius=2.0)
    gf.create_source(world, attached_to="g1", magnitude=10.0)
    result = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    U1 = result.electrode_potentials["g1"][0].real
    U2 = result.electrode_potentials["g2"][0].real
    assert 0.0 < U2 < U1
    # At an 8 m separation, typically < 10 % of the source potential.
    assert U2 / U1 < 0.10


def test_image_no_conductor_no_current_in_g2() -> None:
    """Without a connection, only the source electrode carries current."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_electrode(world, "ring", name="g2",
                        center=(8.0, 0, 0.8), radius=2.0)
    gf.create_source(world, attached_to="g1", magnitude=10.0)
    res = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    assert res.electrode_currents["g1"][0] == pytest.approx(10 + 0j)
    assert res.electrode_currents["g2"][0] == pytest.approx(0 + 0j)
    # Yet the induced potential is positive.
    assert res.electrode_potentials["g2"][0].real > 0.0


def test_image_conductor_creates_cluster_and_splits_current() -> None:
    """With a connection: shared cluster, split current, equal potentials."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    g1 = gf.create_electrode(world, "rod", name="g1",
                             position=(0, 0, 0.5), length=1.5)
    g2 = gf.create_electrode(world, "ring", name="g2",
                             center=(8.0, 0, 0.8), radius=2.0)
    gf.create_conductor(world, name="l1", start=g1, end=g2)
    gf.create_source(world, attached_to=g1, magnitude=10.0)

    res = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    # Cluster
    assert sorted(res.clusters["g1"]) == ["g1", "g2"]
    assert sorted(res.clusters["g2"]) == ["g1", "g2"]

    I1 = res.electrode_currents["g1"][0]
    I2 = res.electrode_currents["g2"][0]
    # Both carry current
    assert I1.real > 0.5
    assert I2.real > 0.5
    # Sum of currents equals the source current
    assert (I1 + I2).real == pytest.approx(10.0)

    # Potentials equal (ideal connection)
    U1 = res.electrode_potentials["g1"][0]
    U2 = res.electrode_potentials["g2"][0]
    assert abs(U1 - U2) < 1e-9


def test_image_cluster_impedance_lower_than_single() -> None:
    """The connected cluster must have a lower impedance than the rod alone."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="image", segment_length=0.05)

    # Rod alone
    w_solo = gf.create_world(soil=soil)
    gf.create_electrode(w_solo, "rod", name="g1",
                        position=(0, 0, 0.5), length=1.5)
    gf.create_source(w_solo, attached_to="g1", magnitude=1.0)
    Z_solo = eng.solve(w_solo).cluster_impedance("g1")[0].real

    # Rod + ring connected
    w_pair = gf.create_world(soil=soil)
    g1 = gf.create_electrode(w_pair, "rod", name="g1",
                             position=(0, 0, 0.5), length=1.5)
    g2 = gf.create_electrode(w_pair, "ring", name="g2",
                             center=(8.0, 0, 0.8), radius=2.0)
    gf.create_conductor(w_pair, name="l1", start=g1, end=g2)
    gf.create_source(w_pair, attached_to=g1, magnitude=1.0)
    Z_pair = eng.solve(w_pair).cluster_impedance("g1")[0].real

    # Parallel combination must be smaller.
    assert Z_pair < Z_solo
    # But not arbitrarily small — > 5 % of Z_solo (sanity).
    assert Z_pair > 0.05 * Z_solo


def test_image_cluster_impedance_solo_equals_grounding_impedance() -> None:
    """For a single-electrode cluster both impedances coincide."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=2.0)
    res = gf.create_engine(backend="image", segment_length=0.05).solve(world)

    Z_g = res.grounding_impedance("g1")[0]
    Z_c = res.cluster_impedance("g1")[0]
    assert abs(Z_g - Z_c) < 1e-12


def test_conductor_resolves_electrode_names() -> None:
    """``create_conductor`` records the electrode names for the cluster logic."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    g1 = gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.0)
    g2 = gf.create_electrode(world, "rod", name="g2", position=(5, 0, 0.5), length=1.0)
    cond = gf.create_conductor(world, start=g1, end="g2")
    assert cond.start_electrode == "g1"
    assert cond.end_electrode == "g2"

    # Purely geometric conductor: no electrode names
    cond2 = gf.create_conductor(world, start=(0, 0, -3.0), end=(5, 0, -3.0))
    assert cond2.start_electrode is None
    assert cond2.end_electrode is None


def test_image_engine_auto_routes_two_layer_soil() -> None:
    """``backend='image'`` with TwoLayerSoil switches to image_2layer.

    Earlier the backend raised a ``TypeError`` in this case. With the
    auto-dispatch (see ADR-0001), the engine transparently uses the
    2-layer backend.
    """
    soil = gf.TwoLayerSoil(rho_1=100, rho_2=500, h_1=2.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.0), length=1.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    res = gf.create_engine(backend="image").solve(world)
    assert res.backend == "image_2layer"


def test_solve_image_directly_rejects_non_homogeneous() -> None:
    """``solve_image`` (private function) stays strict: HomogeneousSoil only.

    The friendly auto-dispatch happens only inside :meth:`Engine.solve`.
    """
    from groundfield.solver.image import solve_image

    soil = gf.TwoLayerSoil(rho_1=100, rho_2=500, h_1=2.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.0), length=1.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image")
    with pytest.raises(TypeError, match="HomogeneousSoil"):
        solve_image(world, eng)


# ---------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------


def _make_demo_result() -> tuple[gf.World, gf.FieldResult]:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_electrode(world, "ring", name="g2",
                        center=(8.0, 0, 0.8), radius=2.0)
    gf.create_source(world, attached_to="g1", magnitude=10.0)
    eng = gf.create_engine(backend="image", segment_length=0.1)
    return world, eng.solve(world)


def test_plot_potential_contour_xy_runs() -> None:
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_potential_contour(
        result, world=world, plane="xy", z=0.0,
        extent=(-5, 15, -5, 5), n=40,
    )
    assert fig is not None and len(fig.axes) >= 1


def test_plot_potential_contour_xz_runs() -> None:
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_potential_contour(
        result, world=world, plane="xz", y=0.0,
        extent=(-5, 15, 0, 4), n=40,
    )
    assert fig is not None


def test_plot_potential_radial_runs() -> None:
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_potential_radial(
        result, around="g1", world=world,
        r_max=20.0, n=80, depths=[0.0, 0.5, 1.5],
    )
    # Expect: one line per depth.
    assert len(fig.axes[0].lines) == 3


def test_plot_potential_profile_runs() -> None:
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_potential_profile(
        result, start=(-2.0, 0.0, 0.0), direction=(1, 0, 0),
        distance=15.0, n=50, depths=[0.0, 1.0],
    )
    assert len(fig.axes[0].lines) == 2


# ---------------------------------------------------------------------
# Surface-potential plot
# ---------------------------------------------------------------------


def test_world_bounds_xy_covers_all_electrodes() -> None:
    """The bounding box must contain every electrode footprint."""
    world, _ = _make_demo_result()
    x_min, x_max, y_min, y_max = gf.world_bounds_xy(world)
    # Rod 'g1' at (0, 0, …); ring 'g2' centred at (8, 0) with radius 2.
    assert x_min <= 0.0
    assert x_max >= 10.0  # 8 + radius 2
    assert y_min <= -2.0
    assert y_max >= 2.0


def test_plot_surface_potential_runs_and_uses_world_bounds() -> None:
    """Default extent comes from the world bounds + padding."""
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_surface_potential(
        result, world,
        z=0.0, padding_m=10.0, n=40,
    )
    assert fig is not None
    ax = fig.axes[0]
    # The plot's data extent must cover the world bounds + padding.
    x_min_w, x_max_w, y_min_w, y_max_w = gf.world_bounds_xy(world)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    assert x_lim[0] <= x_min_w - 9.5  # allow tiny float slop
    assert x_lim[1] >= x_max_w + 9.5
    assert y_lim[0] <= y_min_w - 9.5
    assert y_lim[1] >= y_max_w + 9.5


def test_plot_surface_potential_log_mode_runs() -> None:
    """Log scale path executes without error."""
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_surface_potential(
        result, world,
        z=0.0, padding_m=10.0, n=40, log=True,
    )
    assert fig is not None


def test_plot_surface_potential_explicit_extent_overrides_default() -> None:
    """An explicit ``extent`` argument wins over the world bounds."""
    import matplotlib
    matplotlib.use("Agg")

    world, result = _make_demo_result()
    fig = gf.plot_surface_potential(
        result, world,
        extent=(-100.0, 100.0, -50.0, 50.0), n=40,
    )
    ax = fig.axes[0]
    assert ax.get_xlim()[0] <= -99.0
    assert ax.get_xlim()[1] >= 99.0
