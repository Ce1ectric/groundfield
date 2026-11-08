"""Tests for the experimental FEM (axisymmetric volume PDE) backend.

The FEM backend reduces every cluster to its equivalent hemisphere
and solves the volume PDE in (s, z) coordinates. The reduction is
exact for hemispheres and, by construction, reproduces the Dwight
resistance of a single electrode in homogeneous soil; for clusters,
several clusters and electrodes reaching across a layer interface it is
not a valid cross-check and warns (10.23). ``test_fem_*`` accordingly
checks an enveloped agreement with the closed-form references rather
than bit-exact identity.
"""

from __future__ import annotations

import math

import pytest

import groundfield as gf
from groundfield.references import dwight1936 as dw

SEG = 0.05


def _world_rod(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_fem_rod_matches_dwight() -> None:
    """FEM on a single rod matches Dwight within 10 % (equivalent-hemisphere)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    Z = eng.solve(_world_rod(soil)).cluster_impedance("g1")[0].real
    R_dw = dw.rod(rho=100.0, length=1.5, radius=0.005)
    assert abs(Z - R_dw) / R_dw < 0.10, (
        f"FEM={Z:.3f} vs Dwight={R_dw:.3f}, Δ={abs(Z - R_dw) / R_dw * 100:.2f}%"
    )


def test_fem_two_layer_K_zero_collapses() -> None:
    """For ρ₁ = ρ₂ the FEM result must equal the homogeneous one.

    The FEM mesh follows the soil description (the layer interface is
    cut into the elements), so a 2-layer stack with equal
    resistivities still carries the interface as a mesh feature even
    though the physics is identical to a homogeneous half-space. Since
    0.15.0 the mesh is graded relative to the equivalent hemisphere
    instead of the domain height, so that feature costs only the
    logarithmic resolution loss of the larger truncation radius:
    measured 0.06 % at ``h_1 = 2 m`` (19.5 % before). The full sweep
    over ``h_1`` lives in ``tests/test_pass9_fem.py``.
    """
    rho = 100.0
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    soil_h = gf.HomogeneousSoil(resistivity=rho)
    soil_2 = gf.TwoLayerSoil(rho_1=rho, rho_2=rho, h_1=2.0)
    Z_h = eng.solve(_world_rod(soil_h)).cluster_impedance("g1")[0].real
    Z_2 = eng.solve(_world_rod(soil_2)).cluster_impedance("g1")[0].real
    assert abs(Z_2 - Z_h) / Z_h < 0.01


def test_fem_two_layer_low_rho_below_lowers_R() -> None:
    """A more conductive lower layer must lower R; a more resistive one raises it.

    The FEM backend uses an equivalent-hemisphere reduction; the layer
    contrast is only visible in the result when the equivalent
    hemisphere actually crosses the layer boundary. We pick a thin
    upper layer (``h_1 = 0.3 m``) so that the layer effect dominates.
    """
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    soil_low = gf.TwoLayerSoil(rho_1=100.0, rho_2=10.0, h_1=0.1)
    soil_high = gf.TwoLayerSoil(rho_1=100.0, rho_2=1000.0, h_1=0.1)
    soil_h = gf.HomogeneousSoil(resistivity=100.0)
    Z_low = eng.solve(_world_rod(soil_low)).cluster_impedance("g1")[0].real
    Z_high = eng.solve(_world_rod(soil_high)).cluster_impedance("g1")[0].real
    Z_h = eng.solve(_world_rod(soil_h)).cluster_impedance("g1")[0].real
    assert Z_low < Z_h < Z_high, (Z_low, Z_h, Z_high)


def test_fem_metadata_exposes_a_eq() -> None:
    """The FEM result reports the equivalent-hemisphere radius per cluster."""
    soil = gf.HomogeneousSoil(resistivity=100.0)
    eng = gf.create_engine(backend="fem", segment_length=SEG)
    res = eng.solve(_world_rod(soil))
    assert res.backend == "fem"
    assert "equivalent_hemisphere_radius" in res.metadata
    a_map = res.metadata["equivalent_hemisphere_radius"]
    assert len(a_map) == 1
    a_eq = next(iter(a_map.values()))
    assert 0.05 < a_eq < 1.5  # plausible for a 1.5 m rod


def test_fem_two_rod_cluster() -> None:
    """Two parallel rods connected by a conductor must give a lower R
    than either single rod alone (parallel-electrode rule)."""
    soil = gf.HomogeneousSoil(resistivity=100.0)

    def _solve(world, eng_be):
        eng = gf.create_engine(backend=eng_be, segment_length=SEG)
        return eng.solve(world).cluster_impedance("g1")[0].real

    w_single = gf.create_world(soil=soil)
    gf.create_electrode(w_single, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_source(w_single, attached_to="g1", magnitude=1.0)

    w_pair = gf.create_world(soil=soil)
    gf.create_electrode(w_pair, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
    gf.create_electrode(w_pair, "rod", name="g2", position=(8, 0, 0.0), length=1.5)
    gf.create_conductor(w_pair, name="bond", start="g1", end="g2")
    gf.create_source(w_pair, attached_to="g1", magnitude=1.0)

    Z_single = _solve(w_single, "fem")
    # Two electrodes in one cluster: the reduction sums the hemisphere
    # radii and says so (10.23).
    with pytest.warns(UserWarning, match="sum of their radii"):
        Z_pair = _solve(w_pair, "fem")
    assert Z_pair < Z_single, (Z_pair, Z_single)
    # Two equal hemispheres in parallel: R_pair ≈ R_single / 2.
    # The FEM equivalent-hemisphere reduction places the doubled
    # hemisphere at the cluster centroid; mesh truncation and the
    # geometric reduction together leave a ~20 % envelope around the
    # ideal parallel-hemisphere value.
    assert Z_pair == pytest.approx(Z_single / 2.0, rel=0.25)


# ---------------------------------------------------------------------
# 10.23 — the equivalent-hemisphere reduction says when it is not a
# valid cross-check; the backend is experimental
# ---------------------------------------------------------------------


def _fem_warnings(world) -> tuple[gf.FieldResult, list[str]]:
    """Solve with fem; return the result and the reduction warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = gf.create_engine(backend="fem", segment_length=0.5).solve(world)
    texts = [
        str(w.message) for w in rec if w.category.__name__ == "FemReductionWarning"
    ]
    return res, texts


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "builder", "needle"),
    [
        # ring r = 10 m at 0.8 m over h_1 = 2 m: a_eq = 4.6 m reaches the
        # bottom layer the ring never touches. Over a more resistive layer
        # (30/1000 Ohm m) fem reads +22.7 % against image_2layer and warns;
        # over a more conductive one (1000/30, -92 %) it raises since user
        # decision Q23, see test_fem_rejects_a_more_conductive_layer_*. The
        # opposite case (rod 0-3 m through h_1 = 2 m, +1092 %) raises since
        # the eng2 stream, see test_fem_rejects_an_electrode_below_its_hemisphere.
        (
            gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=2.0),
            lambda w: gf.create_electrode(
                w, "ring", name="g1", center=(0, 0, 0.8), radius=10.0
            ),
            "interface",
        ),
    ],
)
def test_fem_warns_when_electrode_and_hemisphere_straddle_an_interface(
    soil, builder, needle
) -> None:
    w = gf.create_world(soil=soil)
    builder(w)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts, "no FemReductionWarning"
    assert needle in texts[0]
    assert "h = 2" in texts[0]


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("rho_1", "rho_2", "h_1", "top", "length"),
    [
        # rod through h_1 (+1092 % vs mom with 1000/30)
        (1000.0, 30.0, 2.0, 0.0, 3.0),
        # rod through h_1 over a resistive layer (-23 % vs mom)
        (100.0, 1000.0, 2.0, 0.0, 3.0),
        # rod entirely below h_1 (+3138 % vs mom)
        (1000.0, 30.0, 2.0, 3.0, 2.0),
    ],
)
def test_fem_rejects_an_electrode_below_its_hemisphere(
    rho_1, rho_2, h_1, top, length
) -> None:
    """The reduction cannot reach a layer below its hemisphere: raise.

    The equivalent hemisphere injects from the surface down to ``a_eq``
    (0.44 m for a 3 m rod); an electrode that reaches below an interface
    above ``a_eq`` injects into a layer the hemisphere never touches.
    v0.15.0 returned 306.4 Ω (mom: 25.69 Ω) silently, the eng1 stream
    warned; the user decision of 10.23 makes it an error.
    """
    from groundfield.solver.registry import BackendCapabilityError

    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, top), length=length)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(BackendCapabilityError, match="'g1'.*h = 2.*'mom'"):
        eng.solve(w)
    with pytest.raises(ValueError, match="hemisphere"):  # also on a direct call
        from groundfield.solver.fem import solve_fem

        solve_fem(w, eng)


def _a_eq_of_rod(top: float = 0.5, length: float = 3.0) -> float:
    from groundfield.solver.fem import equivalent_hemisphere_radius

    rod = gf.RodElectrode(name="g1", position=(0.0, 0.0, top), length=length)
    return equivalent_hemisphere_radius(rod, 100.0)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("rho_1", "rho_2", "fraction"),
    [
        (1000.0, 30.0, None),  # the auditors' case: h_1 = 0.4 m (+197 % vs mom)
        (1000.0, 30.0, 0.55),  # +51 % vs image_2layer
        (1000.0, 30.0, 1.0),  # h_1 = a_eq: joins the through case
        (30.0, 1000.0, 0.8),  # -85 %
    ],
)
def test_fem_rejects_a_rod_below_a_top_layer_holding_most_of_its_hemisphere(
    rho_1, rho_2, fraction
) -> None:
    """A rod wholly below an interface between a_eq/2 and a_eq: raise.

    The guard of 10.23 compared the interface with the hemisphere radius
    only: a 3 m rod 0.5-3.5 m below h_1 = 0.4 m < a_eq warned and returned
    31.1 Ohm (mom: 10.5 Ohm, +197 %).
    """
    # audit A2E-2
    from groundfield.solver.registry import BackendCapabilityError

    h_1 = 0.4 if fraction is None else fraction * _a_eq_of_rod()
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=3.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(BackendCapabilityError, match="'g1'.*of its equivalent"):
        eng.solve(w)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize("fraction", [0.2, 0.45])
def test_fem_only_warns_below_a_thin_top_layer(fraction: float) -> None:
    """h_1 <= a_eq/2: the hemisphere lies mostly in the rod's layer (+16 %, +38 %)."""
    # audit A2E-2
    h_1 = fraction * _a_eq_of_rod()
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=h_1))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=3.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res, texts = _fem_warnings(w)
    assert texts and "lies above the electrode's lowest point" in texts[0]
    assert math.isfinite(res.electrode_potentials["g1"][0].real)


@pytest.mark.finding("10.23")
def test_fem_guard_applies_to_a_bonded_cluster_wholly_below_the_interface() -> None:
    """Two bonded rods (radii summed): wholly below raises, one above warns."""
    # audit A2E-2
    from groundfield.solver.registry import BackendCapabilityError

    a_sum = 2.0 * _a_eq_of_rod()

    def world(top_b: float) -> gf.World:
        w = gf.create_world(
            soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.7 * a_sum)
        )
        gf.create_electrode(
            w, "rod", name="g1", position=(0, 0, 0.7 * a_sum + 0.1), length=3.0
        )
        gf.create_electrode(w, "rod", name="g2", position=(3, 0, top_b), length=3.0)
        gf.create_conductor(w, start="g1", end="g2", name="bond")
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(BackendCapabilityError, match="every electrode of the cluster"):
        eng.solve(world(0.7 * a_sum + 0.1))
    _, texts = _fem_warnings(world(0.2))  # g2 starts in the top layer
    assert texts


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    "electrode",
    [
        # resistive top 1000/30 Ohm m: fem +22.0 % against image_2layer
        dict(kind="strip", start=(0.0, 0.0, 0.8), end=(10.0, 0.0, 0.8)),
        # resistive top 1000/30 Ohm m: fem +3.1 % against image_2layer
        dict(kind="ring", center=(0.0, 0.0, 0.8), radius=5.0),
    ],
)
def test_fem_warns_when_the_electrode_lies_below_a_thin_top_layer(electrode) -> None:
    """Audit round 1, F01: the hemisphere spans the soil from the surface.

    Strip and ring lie at 0.8 m, entirely below a 0.5 m top layer; their
    hemispheres (a_eq ≈ 1 … 2 m) cross the interface as well, so neither
    the raise (electrode below a hemisphere that ends above the interface)
    nor the old warning (interface between the electrode and a_eq) applied,
    and fem was off without a word. The guard covers every interface above
    the deeper of the electrode's lowest point and ``a_eq``. Under a more
    resistive top layer (1000/30 Ω·m) the result keeps the warning; under
    the audit's more conductive one (30/1000 Ω·m: −70 % and −47 % against
    ``mom``) it raises since user decision Q23
    (:func:`test_fem_rejects_a_more_conductive_layer_its_hemisphere_spans`).
    """
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.5))
    params = dict(electrode)
    gf.create_electrode(w, params.pop("kind"), name="g1", **params)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res, texts = _fem_warnings(w)
    assert math.isfinite(res.cluster_impedance("g1")[0].real)
    assert texts, "no FemReductionWarning"
    assert "interface" in texts[0] and "h = 0.5" in texts[0], texts[0]


# ---------------------------------------------------------------------
# User decision Q23 — a more conductive layer that the hemisphere spans and
# no electrode of the cluster touches is an error
# ---------------------------------------------------------------------


def _rod_world(soil, top: float = 0.5, length: float = 3.0) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, top), length=length)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _electrode_world(soil, kind: str, **params) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, kind, name="g1", **params)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _three_layers(rhos, thicknesses) -> gf.MultiLayerSoil:
    return gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(rho=rho, thickness=h)
            for rho, h in zip(rhos, [*thicknesses, None], strict=True)
        ]
    )


_A_ROD = 0.44231  # equivalent-hemisphere radius of a 3 m rod, r = 5 mm (m)
_A_RING10 = 4.622583619251554  # a_eq of a ring r = 10 m at 0.8 m, r = 5 mm (m)

_STRIP = dict(start=(0.0, 0.0, 0.8), end=(10.0, 0.0, 0.8))


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("world", "layer", "remedy"),
    [
        # rod 0.5-3.5 m under a conductive top layer h_1 = 0.2 a_eq: the base
        # warned and returned -38.2 % / -37.2 % (image_2layer / mom)
        pytest.param(
            lambda: _rod_world(gf.TwoLayerSoil(rho_1=100, rho_2=1000, h_1=0.0885)),
            "0 ... 0.0885 m (rho = 100 Ohm m)",
            "'image_2layer'",
            id="rod-100/1000-thin-top",
        ),
        # a 22 mm top layer is enough: -80.6 % / -80.3 % on the base
        pytest.param(
            lambda: _rod_world(gf.TwoLayerSoil(rho_1=30, rho_2=10000, h_1=0.0221)),
            "0 ... 0.0221 m (rho = 30 Ohm m)",
            "'image_2layer'",
            id="rod-30/10000-22mm-top",
        ),
        # audit F01 strip at 0.8 m under h_1 = 0.5 m: -70.9 % / -70.3 %
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=30, rho_2=1000, h_1=0.5), "strip", **_STRIP
            ),
            "0 ... 0.5 m (rho = 30 Ohm m)",
            "'image_2layer'",
            id="strip-30/1000-F01",
        ),
        # audit F01 ring r = 5 m at 0.8 m under h_1 = 0.5 m: -48.1 % / -47.2 %
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=30, rho_2=1000, h_1=0.5),
                "ring",
                center=(0, 0, 0.8),
                radius=5.0,
            ),
            "0 ... 0.5 m (rho = 30 Ohm m)",
            "'image_2layer'",
            id="ring5-30/1000-F01",
        ),
        # ring r = 10 m at 0.8 m above a conductive lower layer at 2 m
        # (a_eq = 4.62 m): -92.1 % / -92.0 % on the base
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=1000, rho_2=30, h_1=2.0),
                "ring",
                center=(0, 0, 0.8),
                radius=10.0,
            ),
            "below 2 m (rho = 30 Ohm m)",
            "'image_2layer'",
            id="ring10-above-1000/30",
        ),
        # structural: a near-equal contrast raises too (-2.5 % / -1.4 %)
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=120, rho_2=100, h_1=1.0),
                "ring",
                center=(0, 0, 0.8),
                radius=2.0,
            ),
            "below 1 m (rho = 100 Ohm m)",
            "'image_2layer'",
            id="ring2-above-120/100",
        ),
        # three layers, ring in the top layer: the middle layer is spanned
        # (a_eq = 4.62 m > 2 m); mom_sommerfeld solves the world
        pytest.param(
            lambda: _electrode_world(
                _three_layers((1000, 30, 1000), (2.0, 3.0)),
                "ring",
                center=(0, 0, 0.8),
                radius=10.0,
            ),
            "2 ... 5 m (rho = 30 Ohm m)",
            "'mom_sommerfeld'",
            id="ring10-three-layers",
        ),
        # three layers, rod in the second layer: no backend solves it yet
        pytest.param(
            lambda: _rod_world(_three_layers((30, 1000, 100), (0.1, 5.0))),
            "0 ... 0.1 m (rho = 30 Ohm m)",
            "No backend solves",
            id="rod-three-layers-second-layer",
        ),
    ],
)
def test_fem_rejects_a_more_conductive_layer_its_hemisphere_spans(
    world, layer, remedy
) -> None:
    """The hemisphere feeds a better-conducting layer the electrodes never touch.

    ``fem`` replaces the cluster by a hemisphere from the surface down to
    ``a_eq``. Where that hemisphere spans a layer that no electrode of the
    cluster touches and that conducts better than every layer they touch,
    the reduction injects the current there and reads far too low
    (non-conservative). Up to bbaea9f this was a ``FemReductionWarning``
    (never silent: every such world warned, audit D2A-05); the user decided
    on an error, independent of the contrast.
    """
    # user decision Q23
    from groundfield.solver.fem import solve_fem
    from groundfield.solver.registry import BackendCapabilityError

    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(BackendCapabilityError) as info:
        eng.solve(world())
    text = str(info.value)
    assert "cluster {'g1'}" in text, text
    assert f"spans the layer {layer}" in text, text
    assert "a_eq = " in text and "none of its electrodes touches" in text, text
    assert "against 'image_2layer' and 'mom'" in text, text  # measured range
    assert remedy in text, text
    with pytest.raises(ValueError, match="none of its electrodes touches"):
        solve_fem(world(), eng)  # the direct call raises the same


@pytest.mark.finding("10.23")
def test_fem_q23_message_names_the_electrodes_resistivity() -> None:
    """Cluster, layer, the electrodes' resistivity and a_eq are in the text."""
    # user decision Q23
    w = _rod_world(gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.2 * _A_ROD))
    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(gf.BackendCapabilityError) as info:
        eng.solve(w)
    text = str(info.value)
    assert "rho = 30 Ohm m" in text and "they touch (rho = 1000 Ohm m)" in text
    assert "a_eq = 0.442 m" in text
    assert "-97 %" in text  # the measured error range reaches -97 %


def _ring10_with_bare_conductor(soil, z_conductor: float) -> gf.World:
    """Ring r = 10 m at 0.8 m bonded to a rod by a bare galvanic conductor."""
    from groundfield.conductors import Conductor

    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=10.0)
    gf.create_electrode(w, "rod", name="g2", position=(60, 0, 0.5), length=1.0)
    w.add_conductor(
        Conductor(
            name="bare",
            start=(10, 0, z_conductor),
            end=(60, 0, z_conductor),
            start_electrode="g1",
            end_electrode="g2",
            cross_section=50e-6,
            discretize_segment_length=5.0,
            coupling_to_soil="galvanic",
        )
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _fem_message(world, segment_length: float = 0.5) -> str:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(gf.BackendCapabilityError) as info:
            gf.create_engine(backend="fem", segment_length=segment_length).solve(world)
    return str(info.value)


def _solves(backend: str, world, segment_length: float = 0.5) -> bool:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(backend=backend, segment_length=segment_length).solve(
            world
        )
    return math.isfinite(res.cluster_impedance("g1")[0].real)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("world", "named", "refusing"),
    [
        # every segment 5 cm above h_1: mom's point form refuses (closer than
        # 0.387 segment lengths), the line kernel of image_2layer does not
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.85),
                "ring",
                center=(0, 0, 0.8),
                radius=10.0,
            ),
            ("image_2layer",),
            ("mom",),
            id="two-layers-mom-refuses",
        ),
        # a bare conductor below h_1 = 2 m in three layers: mom_sommerfeld
        # refuses the cross-layer geometry (audit D2A-06)
        pytest.param(
            lambda: _ring10_with_bare_conductor(
                _three_layers((1000.0, 30.0, 1000.0), (2.0, 3.0)), 2.5
            ),
            (),
            ("mom_sommerfeld",),
            id="three-layers-bare-conductor-below-h1",
        ),
        # a bare conductor in the air: every backend refuses its segments
        pytest.param(
            lambda: _ring10_with_bare_conductor(
                gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0), -0.5
            ),
            (),
            ("mom", "image_2layer"),
            id="two-layers-bare-conductor-in-the-air",
        ),
        # the plain case: both are named, both solve
        pytest.param(
            lambda: _electrode_world(
                gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
                "ring",
                center=(0, 0, 0.8),
                radius=10.0,
            ),
            ("mom", "image_2layer"),
            (),
            id="two-layers-both-solve",
        ),
    ],
)
def test_fem_names_only_a_backend_that_solves_the_world(world, named, refusing) -> None:
    """The remedy of the error message passes the named backend's own checks.

    The message named ``'mom'`` / ``'image_2layer'`` (two layers) or
    ``'mom_sommerfeld'`` (three or more, when the electrodes lie in the top
    layer) by a rule of thumb; those backends refused some of these worlds
    — a bare conductor below h_1 in three layers, segments near the
    interface (point form), a conductor in the air (audit D2A-06). The
    message now names a backend only if its capability and geometry checks
    accept the world, and says that no backend solves it otherwise.
    """
    # audit D2A-06
    text = _fem_message(world())
    remedy = text[max(text.rfind("Use backend="), text.rfind("No backend solves")) :]
    for backend in named:
        assert f"'{backend}'" in remedy, remedy
        assert _solves(backend, world())
    for backend in refusing:
        assert f"backend='{backend}'" not in remedy, remedy
        assert f"'{backend}'" in remedy and "this world: " in remedy, remedy
    if not named:
        assert "No backend solves this world yet" in remedy, remedy


@pytest.mark.finding("10.23")
def test_fem_remedy_is_not_refused_by_the_engines_resource_limit() -> None:
    """The resource limit is a setting of the engine, not of the world.

    A fem engine with ``max_segments=50`` makes the dry check of 'mom' and
    'image_2layer' stop at their resource limit; the message must still
    name them (with the limit as a remedy), not claim that no backend
    solves the world.
    """
    # audit D2A-06
    w = _electrode_world(
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
        "ring",
        center=(0, 0, 0.8),
        radius=10.0,
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(gf.BackendCapabilityError) as info:
            gf.Engine(backend="fem", segment_length=0.5, max_segments=50).solve(w)
    text = str(info.value)
    assert "No backend solves" not in text, text
    assert "Use backend='mom' (one current per segment) or 'image_2layer'" in text, text
    assert "Engine.max_segments" in text, text


@pytest.mark.finding("10.23")
def test_fem_q23_uses_the_canonical_layer_stack() -> None:
    """Equal resistivities are no interface: the merged top layer is named."""
    # user decision Q23
    soil = _three_layers((30.0, 30.0, 1000.0), (0.05, 0.05))
    with pytest.raises(gf.BackendCapabilityError, match=r"0 \.\.\. 0\.1 m \(rho = 30"):
        gf.create_engine(backend="fem", segment_length=0.5).solve(_rod_world(soil))


@pytest.mark.finding("10.23")
def test_fem_an_electrode_on_the_interface_touches_both_layers() -> None:
    """A ring lying exactly on h_1 touches the conductive layer: no error.

    Ring r = 5 m (a_eq = 2.57 m) under a 30/1000 Ω·m top layer of 0.5 m:
    at z = 0.5 m it lies on the interface and touches both layers (the
    reduction then only warns, +23.4 % against image_2layer); 7 mm deeper
    it touches the lower layer alone and the spanned top layer, more
    conductive, raises. So does the ring within its wire radius below the
    interface since audit R2A-02, see
    :func:`test_fem_a_wire_within_its_radius_below_a_conductive_layer_raises`.
    """
    # user decision Q23
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.5)
    on = _electrode_world(soil, "ring", center=(0, 0, 0.5), radius=5.0)
    res, texts = _fem_warnings(on)
    assert math.isfinite(res.cluster_impedance("g1")[0].real)
    assert texts and "interface" in texts[0]
    below = _electrode_world(soil, "ring", center=(0, 0, 0.5 + 7e-3), radius=5.0)
    with pytest.raises(gf.BackendCapabilityError, match="none of its electrodes"):
        gf.create_engine(backend="fem", segment_length=0.5).solve(below)


# ---------------------------------------------------------------------
# One contact predicate for the layer rules (audit D2A-01, D2A-02): a
# horizontal wire touches every layer within its wire radius of its axis, a
# rod the layers its axis passes through (an end on an interface does not
# count: the thin-wire model has no end face)
# ---------------------------------------------------------------------


def _fem_verdict(world) -> str:
    """'raise' (a layer rule of the reduction), 'warn' or 'silent'."""
    try:
        _, texts = _fem_warnings(world)
    except gf.BackendCapabilityError:
        return "raise"
    return "warn" if texts else "silent"


@pytest.mark.finding("10.23")
@pytest.mark.parametrize("rho_2", [10000.0, 1000.0])
@pytest.mark.parametrize("gap", [0.0, 1e-3, 3e-3])
def test_fem_a_rod_head_on_the_interface_does_not_touch_the_layer_above(
    rho_2, gap
) -> None:
    """A rod whose head lies on h_1 (or within a radius below it) is below it.

    The thin-wire model has no end face, so the integral engines give a rod
    whose head sits exactly on the interface no contact with the layer
    above. ``fem`` counted the closed interval and only warned for the head
    on h_1 under a conductive 30/10000 (30/1000) top layer of 0.2 a_eq,
    where it reads -92.9 % (-64.7 %) against image_2layer and -92.7 %
    (-63.9 %) against mom, the value the rod 1 um deeper raised for
    (audit D2A-01).
    """
    # audit D2A-01
    h_1 = 0.2 * _A_ROD
    w = _rod_world(gf.TwoLayerSoil(rho_1=30.0, rho_2=rho_2, h_1=h_1), top=h_1 + gap)
    with pytest.raises(gf.BackendCapabilityError, match="none of its electrodes"):
        gf.create_engine(backend="fem", segment_length=0.5).solve(w)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize("reach", [1e-6, 0.02, 0.08])
def test_fem_a_rod_reaching_into_the_conductive_top_layer_keeps_the_warning(
    reach,
) -> None:
    """Both sides: a head that reaches into the layer touches it.

    Then the cluster touches the conductive layer and Q23 does not apply;
    fem is still -77 % / -67 % / -53 % (1 um / 2 cm / 8 cm into a 30/10000
    top layer of 0.2 a_eq) against image_2layer and mom, a warning: a
    share-based contact rule is the author's decision (audit D2A-01).
    """
    # audit D2A-01
    h_1 = 0.2 * _A_ROD
    w = _rod_world(gf.TwoLayerSoil(rho_1=30.0, rho_2=10000.0, h_1=h_1), top=h_1 - reach)
    assert _fem_verdict(w) == "warn"


@pytest.mark.finding("10.23")
@pytest.mark.parametrize("gap", [1e-6, 3e-3, 5e-3, 7e-3])
def test_fem_a_wire_within_its_radius_below_a_conductive_layer_raises(gap) -> None:
    """A ring (wire radius 5 mm) whose axis lies within one radius below h_1.

    The Q23 rule asks for the axis (audit R2A-02): the ring touches the
    1000 Ohm m layer only, and the spanned 30 Ohm m top layer conducts
    better, so it raises. fem is +23.4 % / +9.3 % / +0.1 % / -6.2 % against
    image_2layer and the reduced-kernel ring series there; the message says
    that fem misplaces the current and quotes the measured range with its
    positive end instead of "reads too low". Audit D2A-02 had counted the
    band as contact and made the first three a warning, which also turned
    errors of -30 ... -89 % into warnings
    (:func:`test_fem_partial_contact_with_a_more_conductive_layer_raises`).
    """
    # audit R2A-02
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.5)
    text = _fem_message(
        _electrode_world(soil, "ring", center=(0, 0, 0.5 + gap), radius=5.0)
    )
    assert "none of its electrodes touches" in text, text
    assert "reads too low" not in text and "misplaces the current" in text, text
    assert "+30 %" in text, text  # the measured range ends above +23.4 %


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "radius", "z"),
    [
        # ring 3 mm / 5 mm above a conductive lower layer that the hemisphere
        # spans: -55.6 % / -85.4 % against image_2layer (Q23)
        pytest.param(
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=0.5),
            5.0,
            0.497,
            id="1000/30-3mm-above",
        ),
        pytest.param(
            gf.TwoLayerSoil(rho_1=3000.0, rho_2=30.0, h_1=0.5),
            5.0,
            0.495,
            id="3000/30-5mm-above",
        ),
        pytest.param(
            gf.TwoLayerSoil(rho_1=10000.0, rho_2=30.0, h_1=0.8),
            2.0,
            0.797,
            id="10000/30-3mm-above",
        ),
        # ring 5 mm below a conductive top layer in the A2E-2 band
        # (h_1 = 0.66 a_eq): -79.4 % against image_2layer (A2E-2)
        pytest.param(
            gf.TwoLayerSoil(rho_1=30.0, rho_2=10000.0, h_1=0.8),
            2.0,
            0.805,
            id="30/10000-5mm-below",
        ),
    ],
)
def test_fem_partial_contact_with_a_more_conductive_layer_raises(
    soil, radius, z
) -> None:
    """A wire within its radius of a more conductive layer it does not reach.

    Audit D2A-02 let a horizontal wire touch every layer within its wire
    radius of its axis, for every rule. That lifted the Q23 and A2E-2
    errors where fem is far too low (-30 ... -89 % against image_2layer and
    a radius-regularised ring series, all raised on 563ead7); 4fc8153 only
    warned. The band now serves the 10.23 rule only, which it makes
    stricter (audit R2A-02).
    """
    # audit R2A-02
    assert _fem_verdict(
        _electrode_world(soil, "ring", center=(0, 0, z), radius=radius)
    ) == ("raise")


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(("rho_1", "rho_2"), [(30.0, 1000.0), (1000.0, 30.0)])
def test_fem_a_wire_on_the_interface_is_not_below_it(rho_1, rho_2) -> None:
    """The A2E-2 rule: a ring lying on h_1 is not "below" the interface.

    Ring r = 2 m on h_1 = 0.8 m (a_eq = 1.21 m, so h_1 is in the A2E-2 band
    a_eq/2 < h <= a_eq): it touches both layers, but the rule treated it as
    lying below h_1 and raised "in a layer the electrodes do not touch";
    fem is -0.6 % / +10.2 % against image_2layer and the ring series there
    (audit D2A-02). 7 mm below h_1 it lies below the interface and raises.
    """
    # audit D2A-02
    soil = gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=0.8)
    on = _electrode_world(soil, "ring", center=(0, 0, 0.8), radius=2.0)
    res, texts = _fem_warnings(on)
    assert math.isfinite(res.cluster_impedance("g1")[0].real) and texts
    below = _electrode_world(soil, "ring", center=(0, 0, 0.807), radius=2.0)
    with pytest.raises(gf.BackendCapabilityError, match="of its equivalent hemisphere"):
        gf.create_engine(backend="fem", segment_length=0.5).solve(below)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "kind", "params", "depths"),
    [
        # the second interface is 0.1 + 0.2 = 0.30000000000000004 m
        pytest.param(
            _three_layers((300.0, 1000.0, 30.0), (0.1, 0.2)),
            "ring",
            dict(radius=5.0),
            (0.3, 0.1 + 0.2),
            id="ring",
        ),
        # canonical top layer 0.05 + 0.03 m (30 and 30 Ohm m merge)
        pytest.param(
            _three_layers((30.0, 30.0, 1000.0), (0.05, 0.03)),
            "rod",
            dict(length=3.0),
            (0.08, 0.05 + 0.03),
            id="rod-head",
        ),
    ],
)
def test_fem_contact_does_not_depend_on_float_sums(soil, kind, params, depths) -> None:
    """An electrode written at 0.3 and one at 0.1 + 0.2 get the same verdict."""
    # audit D2A-02

    def verdict(z: float) -> str:
        key = "center" if kind == "ring" else "position"
        return _fem_verdict(_electrode_world(soil, kind, **{key: (0, 0, z)}, **params))

    assert verdict(depths[0]) == verdict(depths[1])


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(("rho_1", "rho_2"), [(1000.0, 30.0), (30.0, 1000.0)])
@pytest.mark.parametrize("gap", [0.0, 3e-3])
def test_fem_a_wire_touching_a_layer_below_its_hemisphere_raises(
    rho_1, rho_2, gap
) -> None:
    """The 10.23 rule with the same predicate: a wire on (or within one radius
    above) an interface below a_eq touches the layer its hemisphere does not
    reach.

    Ring r = 2 m at 2 m - gap over h_1 = 2 m (a_eq = 1.3 m): +1008 % / +487 %
    (1000/30) and -18 % / -17 % (30/1000) against image_2layer and the ring
    series, silent or only a warning on 563ead7 (the axis lies above the
    interface).
    """
    # audit D2A-02
    w = _electrode_world(
        gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=2.0),
        "ring",
        center=(0, 0, 2.0 - gap),
        radius=2.0,
    )
    with pytest.raises(gf.BackendCapabilityError, match="hemisphere of its cluster"):
        gf.create_engine(backend="fem", segment_length=0.5).solve(w)


# ---------------------------------------------------------------------
# Audit R2A-01: round-off in the end depths of a strip. StripElectrode
# accepts end depths that differ by up to 1e-9 m as horizontal; only a
# rod is non-horizontal, every other electrode has one depth.
# ---------------------------------------------------------------------


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("start_z", "end_z"),
    [
        pytest.param(0.1 + 0.2, 0.3, id="float-sum"),
        pytest.param(0.3, 0.3 + 1e-10, id="1e-10-at-the-end"),
        pytest.param(0.3 + 1e-10, 0.3, id="1e-10-at-the-start"),
    ],
)
def test_fem_round_off_in_a_strip_does_not_change_the_verdict(start_z, end_z) -> None:
    """A strip lying in the interface h_1 = 0.3 m warns like the exact one.

    fem took a strip whose end depths differ by round-off for a rod, so the
    rod's contact rule (the open interval between its ends) applied: the
    ends 0.1 + 0.2 / 0.3 touched no layer and raised a bare ``ValueError:
    min() iterable argument is empty``, the ends 0.3 / 0.3 + 1e-10 raised a
    Q23 error saying that the strip touches neither layer, while the exact
    strip warns (audit R2A-01; 563ead7 warned for all of them).
    """
    # audit R2A-01
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.3)
    exact = _electrode_world(soil, "strip", start=(0, 0, 0.3), end=(10, 0, 0.3))
    noisy = _electrode_world(soil, "strip", start=(0, 0, start_z), end=(10, 0, end_z))
    assert _fem_verdict(exact) == "warn"
    assert _fem_verdict(noisy) == _fem_verdict(exact)


@pytest.mark.finding("10.23")
def test_fem_round_off_in_a_strip_below_the_interface_does_not_change_the_verdict() -> (
    None
):
    """A strip 3 mm below h_1 = 0.5 m with 1e-10 m of round-off in one end.

    4fc8153 warned for the exact strip (the wire-radius band) and raised
    Q23 for the noisy one (taken for a rod, audit R2A-01).
    """
    # audit R2A-01
    soil = gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=0.5)
    exact = _electrode_world(soil, "strip", start=(0, 0, 0.503), end=(10, 0, 0.503))
    noisy = _electrode_world(
        soil, "strip", start=(0, 0, 0.503), end=(10, 0, 0.503 + 1e-10)
    )
    assert _fem_verdict(noisy) == _fem_verdict(exact)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("start_z", "end_z", "needle"),
    [
        # 4fc8153 solved it (24.6326 Ohm); image and mom refuse it
        pytest.param(1e-3, 1e-3 + 1e-10, "soil-surface plane", id="surface-plane"),
        pytest.param(0.0, 1e-10, "at or above the soil surface", id="at-z=0"),
        # 4fc8153: ValueError: min() iterable argument is empty
        pytest.param(-1e-10, 0.0, "at or above the soil surface", id="above-z=0"),
    ],
)
def test_fem_round_off_does_not_bypass_the_out_of_soil_refusal(
    start_z, end_z, needle
) -> None:
    """fem refuses a strip in or above the surface plane like image and mom."""
    # audit R2A-01
    import warnings

    w = _electrode_world(
        gf.HomogeneousSoil(rho=100.0),
        "strip",
        start=(0, 0, start_z),
        end=(10, 0, end_z),
    )
    for backend in ("image", "mom"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="soil.surface"):
                gf.create_engine(backend=backend, segment_length=0.5).solve(w)
    with pytest.raises(ValueError, match=needle) as info:
        _fem_warnings(w)
    assert not isinstance(info.value, gf.BackendCapabilityError), info.value


def _documented(pattern: str) -> list[str]:
    """Where the fem docs (engine page, FemReductionWarning) miss ``pattern``."""
    import re
    from pathlib import Path

    from groundfield.solver.fem import FemReductionWarning

    page = Path(__file__).resolve().parents[1] / "docs/explanation/engines/fem.md"
    texts = {
        "fem.md": page.read_text(encoding="utf-8"),
        "FemReductionWarning": FemReductionWarning.__doc__,
    }
    return [k for k, t in texts.items() if not re.search(pattern, " ".join(t.split()))]


def _z_ref(backend: str, world) -> float:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(backend=backend, segment_length=0.25).solve(world)
    return res.cluster_impedance("g1")[0].real


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "kind", "params"),
    [
        # ring r = 10 m over 30/10000 at 2 m: -1.0 % / -0.9 %
        (
            gf.TwoLayerSoil(rho_1=30.0, rho_2=10000.0, h_1=2.0),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
        ),
        # 3 m rod under 10000/30 at 0.49 a_eq: +44.7 % / +46.9 % (+47.6 % at
        # half the segment length)
        (
            gf.TwoLayerSoil(rho_1=10000.0, rho_2=30.0, h_1=0.49 * _A_ROD),
            "rod",
            dict(position=(0, 0, 0.5), length=3.0),
        ),
    ],
)
def test_fem_documented_range_of_the_resistive_warning_holds(
    soil, kind, params
) -> None:
    """A more resistive spanned layer warns at -1 ... +48 %, as documented.

    The docs quoted +3 ... +45 %; the extremes measured by the audit are
    -1.0 % (ring over 30/10000) and +47.6 % (rod under 10000/30, mom at half
    the segment length) (audit D2A-04).
    """
    # audit D2A-04
    w = _electrode_world(soil, kind, **params)
    res, texts = _fem_warnings(w)
    assert texts, "no FemReductionWarning"
    fem = res.cluster_impedance("g1")[0].real
    for backend in ("image_2layer", "mom"):
        err = 100.0 * (fem / _z_ref(backend, w) - 1.0)
        assert -1.5 <= err <= 48.5, f"{backend}: {err:+.1f} %"
    assert _documented(r"−1 (% )?… \+48 %") == []


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    ("rho_1", "rho_2", "h_1"),
    [
        (30.0, 10000.0, 9.99),  # -15.5 % / -15.2 %
        (100.0, 1000.0, _A_RING10 * (1 + 1e-9)),  # +15.2 % / +15.6 %
    ],
)
def test_fem_documented_range_of_the_wide_warning_holds(rho_1, rho_2, h_1) -> None:
    """The warning that remains within the half-extent: -16 ... +16 %.

    Since the conductive "wide" worlds raise (D2A-03), the warning covers
    layers that conduct no better than the electrodes'. The docs quoted
    -11 ... -47 % for the conductive ones and +9 % for one resistive case;
    ring r = 10 m at 0.8 m over 30/10000 and 100/1000 Ohm m reads
    -15.5 % ... +15.6 % against image_2layer and mom (audit D2A-03).
    """
    # audit D2A-03
    w = _electrode_world(
        gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1),
        "ring",
        center=(0, 0, 0.8),
        radius=10.0,
    )
    res, texts = _fem_warnings(w)
    assert texts and "half-extent" in texts[0], texts
    fem = res.cluster_impedance("g1")[0].real
    for backend in ("image_2layer", "mom"):
        err = 100.0 * (fem / _z_ref(backend, w) - 1.0)
        assert -16.5 <= err <= 16.5, f"{backend}: {err:+.1f} %"
    assert _documented(r"−16 (% )?… \+16 %") == []


@pytest.mark.finding("10.23")
def test_fem_q23_counts_every_electrode_of_a_bonded_cluster() -> None:
    """Two bonded rods: both below the conductive top layer raise; one in it warns."""
    # user decision Q23
    h_1 = 0.4 * 2.0 * _A_ROD  # below a_sum/2: the A2E-2 rule does not apply

    def world(top_b: float) -> gf.World:
        w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=h_1))
        gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=3.0)
        gf.create_electrode(w, "rod", name="g2", position=(3, 0, top_b), length=3.0)
        gf.create_conductor(w, start="g1", end="g2", name="bond")
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    eng = gf.create_engine(backend="fem", segment_length=0.5)
    with pytest.raises(gf.BackendCapabilityError, match=r"cluster \{'g1', 'g2'\}"):
        eng.solve(world(0.5))
    res, texts = _fem_warnings(world(0.1))  # g2 starts in the top layer
    assert math.isfinite(res.cluster_impedance("g1")[0].real)
    assert texts and "sum of their radii" in texts[0]


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("rho_1", "rho_2", "h_1", "kind", "params"),
    [
        # more resistive top layer over the rod: +15.5 % (image_2layer)
        (1000.0, 30.0, 0.2 * _A_ROD, "rod", dict(position=(0, 0, 0.5), length=3.0)),
        # more resistive lower layer under the ring: +22.7 %
        (30.0, 1000.0, 2.0, "ring", dict(center=(0, 0, 0.8), radius=10.0)),
        # the rod passes through the conductive layer itself: +19.2 %
        (100.0, 10.0, 0.1, "rod", dict(position=(0, 0, 0.0), length=1.5)),
    ],
)
def test_fem_q23_keeps_the_warning_where_the_layer_is_not_more_conductive(
    rho_1, rho_2, h_1, kind, params
) -> None:
    """Both sides of the rule: these spanned layers are not a Q23 error."""
    # user decision Q23
    w = _electrode_world(
        gf.TwoLayerSoil(rho_1=rho_1, rho_2=rho_2, h_1=h_1), kind, **params
    )
    res, texts = _fem_warnings(w)
    assert math.isfinite(res.cluster_impedance("g1")[0].real)
    assert texts, "no FemReductionWarning"


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    ("soil", "kind", "params", "layer"),
    [
        # ring r = 10 m at 0.8 m (a_eq = 4.62 m, half-extent 10 m) over a more
        # conductive layer below its hemisphere: -67.9 % at h_1 = a_eq (1 +
        # 1e-9), the value the world at a_eq (1 - 1e-9) raised for (D2A-03),
        # -46.9 % at 5 m, -11.0 % at 9 m (image_2layer; mom within 1 point)
        pytest.param(
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=_A_RING10 * (1 + 1e-9)),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
            "below 4.62 m (rho = 30 Ohm m)",
            id="ring10-1000/30-just-below-a_eq",
        ),
        pytest.param(
            gf.TwoLayerSoil(rho_1=3000.0, rho_2=30.0, h_1=_A_RING10 * (1 + 1e-9)),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
            "below 4.62 m (rho = 30 Ohm m)",
            id="ring10-3000/30-just-below-a_eq",
        ),
        pytest.param(
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
            "below 5 m (rho = 30 Ohm m)",
            id="ring10-1000/30-5m",
        ),
        pytest.param(
            gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=9.0),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
            "below 9 m (rho = 30 Ohm m)",
            id="ring10-1000/30-9m",
        ),
        # 10 m strip at 0.8 m (half-extent 5 m) over 3000/30 at 3 m
        pytest.param(
            gf.TwoLayerSoil(rho_1=3000.0, rho_2=30.0, h_1=3.0),
            "strip",
            dict(start=(0, 0, 0.8), end=(10, 0, 0.8)),
            "below 3 m (rho = 30 Ohm m)",
            id="strip10-3000/30-3m",
        ),
        # three layers: the conductive middle layer lies within the extent
        pytest.param(
            _three_layers((1000.0, 30.0, 1000.0), (6.0, 2.0)),
            "ring",
            dict(center=(0, 0, 0.8), radius=10.0),
            "6 ... 8 m (rho = 30 Ohm m)",
            id="ring10-three-layers",
        ),
    ],
)
def test_fem_rejects_a_more_conductive_layer_within_the_electrode_extent(
    soil, kind, params, layer
) -> None:
    """Q23 extends to the domain of the "wide" rule (D2A-03).

    A layer below the electrode and its hemisphere but within the
    electrode's horizontal half-extent is reached by the electrode's
    current and much less by the compact hemisphere. Over a more
    conductive layer fem then reads too low, continuously with the Q23
    worlds above a_eq (only a warning up to 563ead7; the eng2 evidence e07
    pinned fem 11.79 Ohm against mom 21.89 Ohm for the ring over h_1 = 5 m).
    Coordinator decision, flagged for the author's review: such a layer
    raises like one the hemisphere spans.
    """
    # audit D2A-03
    with pytest.raises(gf.BackendCapabilityError) as info:
        gf.create_engine(backend="fem", segment_length=0.5).solve(
            _electrode_world(soil, kind, **params)
        )
    text = str(info.value)
    assert "none of its electrodes touches" in text, text
    assert f"the layer {layer}" in text and "half-extent" in text, text


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    ("rhos", "thicknesses"),
    [
        # fem +40.0 % / +46.4 % / +23.7 % / +23.6 % against mom_sommerfeld
        # (+39.2 / +45.8 / +22.7 / +22.6 % against a Hankel-integral ring):
        # the 1000 or 300 Ohm m layer that the hemisphere spans shields both
        pytest.param((100.0, 1000.0, 30.0), (2.0, 4.0), id="100/1000/30-2-4"),
        pytest.param((100.0, 1000.0, 30.0), (1.0, 6.0), id="100/1000/30-1-6"),
        pytest.param((100.0, 300.0, 30.0), (2.0, 5.0), id="100/300/30-2-5"),
        pytest.param((100.0, 1000.0, 30.0), (4.0, 2.0), id="100/1000/30-4-2"),
        # the resistive layer lies below the hemisphere as well: +17.1 % (+16.2 %)
        pytest.param((100.0, 1000.0, 30.0), (5.0, 2.0), id="100/1000/30-5-2"),
    ],
)
def test_fem_a_resistive_layer_in_between_keeps_the_warning(rhos, thicknesses) -> None:
    """Ring r = 10 m at 0.8 m: a 30 Ohm m layer within the half-extent behind
    a more resistive layer.

    Q23 within the half-extent (audit D2A-03) counted the conductive layer
    whatever lay between it and the electrodes, and raised "fem reads too
    low" where fem reads high (563ead7 warned). A layer below the
    hemisphere now counts only if every layer between the electrodes and it
    conducts better than the electrodes' layers (audit R2A-09).
    """
    # audit R2A-09
    w = _electrode_world(
        _three_layers(rhos, thicknesses), "ring", center=(0, 0, 0.8), radius=10.0
    )
    assert _fem_verdict(w) == "warn"


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    ("rhos", "thicknesses", "layers"),
    [
        # both conductive layers lie directly below the ring's layer: -26.5 %
        pytest.param(
            (1000.0, 100.0, 30.0),
            (6.0, 2.0),
            ("6 ... 8 m (rho = 100 Ohm m)", "below 8 m (rho = 30 Ohm m)"),
            id="1000/100/30",
        ),
        # the conductive layer, then a resistive one: -32.7 %
        pytest.param(
            (1000.0, 30.0, 3000.0),
            (5.0, 2.0),
            ("5 ... 7 m (rho = 30 Ohm m)",),
            id="1000/30/3000",
        ),
    ],
)
def test_fem_q23_counts_conductive_layers_directly_below(
    rhos, thicknesses, layers
) -> None:
    """Both sides of the R2A-09 rule: a layer reached through more conductive
    layers only still raises (fem -26.5 % / -32.7 % against mom_sommerfeld)."""
    # audit R2A-09
    text = _fem_message(
        _electrode_world(
            _three_layers(rhos, thicknesses), "ring", center=(0, 0, 0.8), radius=10.0
        )
    )
    for layer in layers:
        assert layer in text, text


@pytest.mark.finding("10.23", "F01")
@pytest.mark.parametrize(
    ("kind", "params", "r_ext"),
    [
        ("ring", dict(center=(0, 0, 0.8), radius=10.0), 10.0),
        ("strip", dict(start=(0, 0, 0.8), end=(10, 0, 0.8)), 5.0),
    ],
)
def test_fem_q23_stops_at_the_horizontal_half_extent(kind, params, r_ext) -> None:
    """Both sides of the new boundary: a layer top at the half-extent.

    Just above the half-extent the conductive layer raises; just below it
    neither rule applies and fem is silent (ring r = 10 m over 1000/30 Ohm m
    at h_1 = 10 m: -7.9 %; 20 m: -0.3 %).
    """

    # audit D2A-03
    def world(h_1: float) -> gf.World:
        soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=h_1)
        return _electrode_world(soil, kind, **params)

    assert _fem_verdict(world(r_ext * (1 - 1e-9))) == "raise"
    assert _fem_verdict(world(r_ext * (1 + 1e-9))) == "silent"


@pytest.mark.finding("10.23", "F01")
def test_fem_warns_when_a_more_resistive_layer_lies_within_the_electrode_extent() -> (
    None
):
    """Ring r = 10 m at 0.8 m over a more resistive layer at h_1 = 5 m.

    The "wide" rule keeps its warning where the layer conducts no better
    than the electrodes' layer: fem +9.2 % against image_2layer (30/1000
    Ohm m), 3.6154 Ohm, unchanged.
    """
    w = _electrode_world(
        gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=5.0),
        "ring",
        center=(0, 0, 0.8),
        radius=10.0,
    )
    res, texts = _fem_warnings(w)
    assert texts, "no FemReductionWarning"
    assert "h = 5 m" in texts[0] and "half-extent" in texts[0], texts[0]
    assert res.cluster_impedance("g1")[0].real == pytest.approx(3.6154, abs=1e-4)


def test_fem_is_silent_for_an_interface_below_the_electrode_extent() -> None:
    """Guard: the same ring over h_1 = 20 m agrees with mom to −0.3 %."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=20.0))
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=10.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts == []


def test_fem_accepts_a_hemisphere_that_reaches_the_electrode_layer() -> None:
    """Guard: a vanishing top layer is no reason to raise (B9 limit, 2e-3).

    A ring at 0.8 m under a 4 mm top layer lies in layer 2, but its
    hemisphere (a_eq = 1.2 m) reaches layer 2 as well; the result tends
    to the homogeneous value of the lower layer. Since audit round 1
    (F01) every interface above the electrode or its hemisphere is
    reported, so this limit case warns although it is accurate — the
    guard is structural, not a measure of the error.
    """
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=4e-3)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res, texts = _fem_warnings(w)
    assert len(texts) == 1 and "h = 0.004" in texts[0], texts
    w_hom = gf.create_world(soil=gf.HomogeneousSoil(rho=30.0))
    gf.create_electrode(w_hom, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_source(w_hom, attached_to="g1", magnitude=1.0)
    ref, _ = _fem_warnings(w_hom)
    z, z_ref = res.cluster_impedance("g1")[0].real, ref.cluster_impedance("g1")[0].real
    assert abs(z / z_ref - 1.0) < 1e-2


def test_fem_does_not_reject_an_equal_resistivity_interface() -> None:
    """Guard: a declared interface between equal resistivities is none (10.48)."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=2.0))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts == []


@pytest.mark.finding("10.23")
def test_fem_warns_for_a_multi_electrode_cluster() -> None:
    """Ring + three rods: radii summed, mutual resistance neglected
    (-35 % against mom in homogeneous soil)."""
    import numpy as np

    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.5), radius=2.0)
    for k, ang in enumerate((0.0, 2.1, 4.2)):
        gf.create_electrode(
            w,
            "rod",
            name=f"r{k}",
            position=(2 * np.cos(ang), 2 * np.sin(ang), 0.5),
            length=4.0,
        )
        gf.create_conductor(w, start="g1", end=f"r{k}")
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts and "4 electrodes" in texts[0]


@pytest.mark.finding("10.23")
def test_fem_warns_when_clusters_are_solved_without_coupling() -> None:
    """A second (passive) cluster reports 0 V and 0 transfer impedance."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p", center=(4.0, 0, 0.8), radius=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    res, texts = _fem_warnings(w)
    assert res.electrode_potentials["p"][0] == 0j
    assert texts and "2 clusters" in texts[0]


@pytest.mark.finding("10.23")
def test_fem_rejects_a_driven_polyline_with_a_capability_error() -> None:
    """A driven electrode without a hemisphere formula is a capability limit.

    The generators build mesh foundations of polyline wires (audit round 2,
    A2G-04); fem raised a bare ``TypeError`` from inside the reduction for
    every generated network. It now raises ``BackendCapabilityError`` before
    meshing, naming the electrode, while a passive polyline still solves.
    """
    import warnings

    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(
        w, "polyline", name="wire", vertices=[(0, 0, 0.7), (5, 0, 0.7), (5, 5, 0.7)]
    )
    gf.create_source(w, attached_to="wire", magnitude=1.0)
    eng = gf.create_engine(backend="fem", frequencies=[50.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(gf.BackendCapabilityError, match="'wire'"):
            w.solve(eng)

    passive = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(passive, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
    gf.create_electrode(
        passive, "polyline", name="wire", vertices=[(20, 0, 0.7), (25, 0, 0.7)]
    )
    gf.create_source(passive, attached_to="g1", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = passive.solve(eng)
    assert res.electrode_potentials["wire"][0] == 0j


_SURFACE_CASES = [
    # (soil, kind, params, needle of the fem message, passive)
    pytest.param(
        gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=2.0),
        "ring",
        dict(center=(0, 0, -0.5), radius=10.0),
        "at or above the soil surface",
        False,
        id="ring-in-the-air-two-layers",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "ring",
        dict(center=(0, 0, 0.0), radius=10.0),
        "at or above the soil surface",
        False,
        id="ring-at-z=0",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "strip",
        dict(start=(0, 0, 0.0), end=(10, 0, 0.0)),
        "at or above the soil surface",
        False,
        id="strip-at-z=0",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "grid_mesh",
        dict(corner=(0, 0, 0.0), size=(10, 10), n_x=2, n_y=2),
        "at or above the soil surface",
        False,
        id="grid-at-z=0",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "mesh",
        dict(corner=(0, 0, -0.7), size=(10, 10)),
        "at or above the soil surface",
        False,
        id="mesh-in-the-air",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "rod",
        dict(position=(0, 0, -0.5), length=1.2),
        "reaches above the soil surface",
        False,
        id="rod-head-in-the-air",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "ring",
        dict(center=(0, 0, 0.0025), radius=10.0),
        "soil-surface plane",
        False,
        id="ring-in-the-surface-plane",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "ring",
        dict(center=(0, 0, -0.5), radius=2.0),
        "at or above the soil surface",
        True,
        id="passive-ring-in-the-air",
    ),
    pytest.param(
        gf.HomogeneousSoil(rho=100.0),
        "polyline",
        dict(vertices=[(0, 5, 0.0), (5, 5, 0.0)]),
        "at or above the soil surface",
        True,
        id="passive-polyline-at-z=0",
    ),
]


def _surface_world(soil, kind: str, params: dict, passive: bool) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, kind, name="g1", **params)
    if passive:  # the probed electrode is passive, a rod is driven
        gf.create_electrode(w, "rod", name="d", position=(30, 0, 0.5), length=3.0)
    gf.create_source(w, attached_to="d" if passive else "g1", magnitude=1.0)
    return w


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("soil", "kind", "params", "needle", "passive"), _SURFACE_CASES
)
def test_fem_rejects_an_electrode_above_the_surface_like_every_backend(
    soil, kind, params, needle, passive
) -> None:
    """An electrode at or above the soil surface is refused by every backend.

    The integral backends refuse a segment whose mid-point lies at or above
    the surface, in the surface plane (``2 z <= wire_radius``) or with a
    part in the air (:func:`groundfield.solver._guards._check_segment_depths`).
    ``fem`` solved such worlds — a ring 0.5 m in the air read like a ring
    1 mm deep, since the hemisphere formulas clamp the depth at 1 mm — and
    a Q23 message named ``'mom'`` / ``'image_2layer'`` for a world both
    refuse (audit D2A-06). It now refuses them as well, before anything
    else, with a plain ``ValueError`` like the others.
    """
    # audit D2A-06
    import warnings

    other = "image" if isinstance(soil, gf.HomogeneousSoil) else "image_2layer"
    for backend in (other, "mom"):  # the rule fem matches
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="soil.surface"):
                gf.create_engine(backend=backend, segment_length=0.5).solve(
                    _surface_world(soil, kind, params, passive)
                )
    world = _surface_world(soil, kind, params, passive)
    with pytest.raises(ValueError, match=needle) as info:
        _fem_warnings(world)
    assert not isinstance(info.value, gf.BackendCapabilityError), info.value
    assert "'g1'" in str(info.value)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("rod", dict(position=(0, 0, 0.0), length=3.0)),  # the driven rod
        ("rod", dict(position=(0, 0, -1e-10), length=3.0)),  # round-off
        ("ring", dict(center=(0, 0, 0.003), radius=10.0)),  # 2 z > a
    ],
)
def test_fem_keeps_electrodes_that_start_at_the_surface(kind, params) -> None:
    """Guard: a rod with its head at z = 0 is the ordinary driven rod."""
    # audit D2A-06
    res, _ = _fem_warnings(
        _surface_world(gf.HomogeneousSoil(rho=100.0), kind, params, False)
    )
    assert math.isfinite(res.cluster_impedance("g1")[0].real)


# ---------------------------------------------------------------------
# The closed forms of the equivalent hemisphere outside their range
# (audit D2A, O-1): a burial depth beyond the electrode's half-extent,
# and the MeshElectrode formula
# ---------------------------------------------------------------------


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("kind", "params", "needle"),
    [
        # 1.88 m strip at 5 m / 9.8 m: fem +369 % / +6120 % against image,
        # silently (audit O-1); Dwight's wire series diverges beyond z = l/2
        ("strip", dict(start=(0, 0, 5.0), end=(1.88, 0, 5.0)), "half its length"),
        ("strip", dict(start=(0, 0, 9.8), end=(1.88, 0, 9.8)), "half its length"),
        # 10 m strip just below z = l/2 = 5 m (+0.1 % ... +3 % up to z = l)
        ("strip", dict(start=(0, 0, 5.01), end=(10, 0, 5.01)), "half its length"),
        # rings deeper than their radius: -5.4 % (r = 0.5 m at 2 m) and
        # -4.6 % (r = 2 m at 8 m) against image, ever lower with depth
        ("ring", dict(center=(0, 0, 2.0), radius=0.5), "its radius"),
        ("ring", dict(center=(0, 0, 8.0), radius=2.0), "its radius"),
    ],
)
def test_fem_rejects_an_electrode_deeper_than_its_formula_reaches(
    kind, params, needle
) -> None:
    """The hemisphere formulas assume a burial depth small against the electrode.

    Dwight's wire formula is the series of the wire and its image in
    s/l (s = 2 z the image distance, l the length) and converges for
    s < l only; his ring formula assumes s << D. ``fem`` used them at any
    depth and read +369 % / +6120 % for a 1.88 m strip at 5 m / 9.8 m in
    homogeneous soil, silently (audit D2A, O-1). It now raises when the
    burial depth exceeds the electrode's horizontal half-extent (half the
    strip length, the ring radius) and names the backends that solve the
    world.
    """
    # audit D2A O-1
    w = _electrode_world(gf.HomogeneousSoil(rho=10.0), kind, **params)
    with pytest.raises(gf.BackendCapabilityError, match=needle) as info:
        gf.create_engine(backend="fem", segment_length=0.25).solve(w)
    text = str(info.value)
    assert "'g1'" in text and "Use backend='image' or 'mom'" in text, text


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("strip", dict(start=(0, 0, 0.8), end=(1.88, 0, 0.8))),  # -0.2 %
        ("strip", dict(start=(0, 0, 5.0), end=(10, 0, 5.0))),  # z = l/2: -0.1 %
        ("ring", dict(center=(0, 0, 2.0), radius=2.0)),  # z = r: +0.7 %
        ("ring", dict(center=(0, 0, 0.8), radius=2.0)),  # benchmark B3
        # Sverak's grid formula stays bounded with depth: +11.2 % at 20 m
        ("grid_mesh", dict(corner=(0, 0, 20.0), size=(20, 20), n_x=4, n_y=4)),
    ],
)
def test_fem_keeps_electrodes_within_their_formula(kind, params) -> None:
    """Guard: within the range the result is unchanged (no error)."""
    # audit D2A O-1
    res, _ = _fem_warnings(
        _electrode_world(gf.HomogeneousSoil(rho=10.0), kind, **params)
    )
    assert math.isfinite(res.cluster_impedance("g1")[0].real)


@pytest.mark.finding("10.23")
def test_fem_rejects_a_driven_mesh_electrode() -> None:
    """MeshElectrode's formula, one strip along the diagonal, misses its wires.

    +118 % ... +193 % against image for 10 m and 20 m meshes at 0.8 m
    (spacing = size ... 1 m), at every depth (audit D2A, O-1); the
    GridMeshElectrode of the same wires reads -1.4 ... +21 % (Sverak). A
    driven mesh raises, a passive one is not reduced and still solves.
    """
    # audit D2A O-1
    import warnings

    w = _electrode_world(
        gf.HomogeneousSoil(rho=100.0), "mesh", corner=(0, 0, 0.8), size=(10, 10)
    )
    with pytest.raises(gf.BackendCapabilityError, match="GridMeshElectrode") as info:
        gf.create_engine(backend="fem", segment_length=0.5).solve(w)
    assert "'g1'" in str(info.value) and "'image'" in str(info.value)
    passive = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(passive, "rod", name="g1", position=(0, 0, 0), length=3.0)
    gf.create_electrode(passive, "mesh", name="m", corner=(20, 0, 0.8), size=(10, 10))
    gf.create_source(passive, attached_to="g1", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.create_engine(backend="fem", segment_length=0.5).solve(passive)
    assert res.electrode_potentials["m"][0] == 0j


@pytest.mark.finding("10.66")
def test_fem_rejects_a_world_without_a_source_like_every_backend() -> None:
    """fem alone solved a source-free world to zeros through Engine.solve.

    Every other backend raised (10.66); since audit round 2 (A2S-05) fem
    raises the same error, and nothing has to be said about clusters.
    """
    # audit A2S-05
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=3.0)
    gf.create_electrode(w, "ring", name="p", center=(4.0, 0, 0.8), radius=1.5)
    with pytest.raises(ValueError, match="no current source"):
        _fem_warnings(w)


@pytest.mark.finding("10.23")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=5.0),
    ],
)
def test_fem_is_silent_for_one_compact_electrode_inside_one_layer(soil) -> None:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    _, texts = _fem_warnings(w)
    assert texts == []


@pytest.mark.finding("10.23")
def test_fem_is_marked_experimental() -> None:
    import groundfield.solver.fem as fem_mod

    assert "experimental" in fem_mod.__doc__.splitlines()[0].lower()
    assert "experimental" in fem_mod.solve_fem.__doc__.strip().splitlines()[0].lower()
    # Claims that the reduction is not entitled to (review pass 10, R2-01).
    assert "forms the third independent cross-check" not in fem_mod.solve_fem.__doc__
    assert "the approximation is appropriate" not in fem_mod.__doc__


# ---------------------------------------------------------------------
# 10.49 — the truncation sphere does not cost near-electrode resolution
# ---------------------------------------------------------------------


@pytest.mark.finding("10.49")
@pytest.mark.parametrize("h_1", [10.0, 100.0, 1000.0, 10000.0])
def test_fem_deep_interface_does_not_degrade_the_resolution(h_1: float) -> None:
    """``TwoLayerSoil(rho, rho, h_1)`` is homogeneous for every h_1.

    The truncation sphere grows with the declared thickness,
    ``r_far = 30 (a_eq + h_1)``; at a fixed radial node count v0.15.0
    lost near-electrode resolution logarithmically: -0.25 % / -0.60 % /
    -1.06 % / -1.62 % at h_1 = 10 m / 100 m / 1 km / 10 km (R2-15).
    """
    eng = gf.create_engine(backend="fem", segment_length=SEG)

    def world(soil):
        w = gf.create_world(soil=soil)
        gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0), length=1.5)
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    r_hom = eng.solve(world(gf.HomogeneousSoil(resistivity=100.0)))
    r_lay = eng.solve(world(gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0, h_1=h_1)))
    z_hom = r_hom.cluster_impedance("g1")[0].real
    z_lay = r_lay.cluster_impedance("g1")[0].real
    assert abs(z_lay / z_hom - 1.0) < 2e-3, (h_1, z_lay, z_hom)
