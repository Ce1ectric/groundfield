"""``Engine.image_far_field``: the form of the far images of the line kernel.

The author's decision of 2026-11-15 (stream fw3xl): the quadrupole far images
of wave 3 (10.28) stay the default; ``image_far_field="point"`` offers
point-form far images for catalogue sweeps where speed matters more than the
last 1e-4: the dense far term of the Tagg/Sunde images and of the images of
the cross-layer correction is left out, their near pairs (within 7.5 segment
lengths, a segment's own images included) stay Galerkin integrals blended
into the point form. Up to audit W3R2A-01 the cross-layer path kept only the
interface image of pairs in one layer in line form, as the base of wave 3,
and long segments below the interface lost up to 58 %. The continuous switch
of user decision D2 and the transverse distance of image pairs (audit W3A-02)
are kept. The option travels through ``Engine`` and the registry, JSON,
``compare_engines`` and ``sweep``, ``FieldResult.potential``
(``metadata["series"]["far_field"]``) and ``solve_mutual_*``; an unknown form
raises ``ValueError`` (audit W3R2A-02).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf

pytestmark = pytest.mark.finding("10.28")


def _houses(soil) -> gf.World:
    """16 house grids (10 x 10 m, 2 x 2 meshes) at 0.8 m, 25 m apart."""
    w = gf.create_world(soil=soil)
    names = []
    for i in range(4):
        for j in range(4):
            name = f"h{i}{j}"
            gf.create_electrode(
                w,
                "grid_mesh",
                name=name,
                corner=(25.0 * i, 25.0 * j, 0.8),
                size=(10.0, 10.0),
                n_x=2,
                n_y=2,
            )
            names.append(name)
    for k, name in enumerate(names[1:], start=1):
        gf.create_conductor(
            w,
            name=f"c{k}",
            start=names[k - 1],
            end=name,
            conductor_type="pen",
            cross_section=50e-6,
            coupling_to_soil="isolated",
        )
    gf.create_source(w, attached_to="h00", magnitude=1.0)
    return w


def _rod_through() -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0))
    gf.create_electrode(
        w, "rod", name="r", position=(0, 0, 0.6), length=6.0, wire_radius=0.008
    )
    gf.create_source(w, attached_to="r", magnitude=1.0)
    return w


def _strip() -> gf.World:
    """The strip of 10.28: 10 m at 0.5 m in 1000/50 Ohm m, h_1 = 1 m."""
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=1000.0, rho_2=50.0, h_1=1.0))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, 0.5), end=(10, 0, 0.5), wire_radius=0.005
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w


def _z(world, anchor, **kw) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.Engine(segment_length=0.5, frequencies=[50.0], **kw).solve(world)
    return float(res.cluster_impedance(anchor)[0].real)


# the author's decision of 2026-11-15
def test_the_default_is_the_quadrupole_form_and_unchanged() -> None:
    """``"quadrupole"`` is the default; requesting it explicitly is the same solve.

    The default is bit-identical to the tree before the option (stream fw3xl,
    evidence ``option_default_vs_pre_option.out``: ten worlds, impedances,
    currents and potentials, numba on and off). The 16 house grids in
    500/100 Ohm m with h_1 = 5 m read 21.0988482 Ohm (34ad269: 21.0980834).
    """
    assert gf.Engine().image_far_field == "quadrupole"
    for world, anchor in (
        (_houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0)), "h00"),
        (_rod_through(), "r"),
    ):
        default = _z(world, anchor)
        assert _z(world, anchor, image_far_field="quadrupole") == default
    z = _z(_houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0)), "h00")
    assert z == pytest.approx(21.098848200945483, rel=1e-12)


#: Re Z of 34ad269 (the base of wave 3, line kernel, ds = 0.5 m), measured
#: with that tree on the author's Mac (evidence
#: ``option_point_vs_base_values.out`` of stream fw3xl).
_BASE = {"houses_500_100_h5": 21.0980834458119, "rod_through_500_100": 39.0164035}


def test_point_reproduces_the_base_where_the_base_is_the_reference() -> None:
    """The point form is the far-image treatment of 34ad269 on the series path.

    16 house grids in 500/100 Ohm m, h_1 = 5 m: every image is at least
    8.4 m (> 7.5 segment lengths) from every segment, so the base used the
    point form for all of them: ``"point"`` +2.4e-7 against the base (the
    blended switch and the radius term of direct pairs), the default +3.6e-5
    (the quadrupole terms). On the cross-layer path the base had the line
    form of the interface image only; the point form keeps the near pairs of
    every image in line form (audit W3R2A-01), and a rod through the
    interface is within 1.2e-5 of the base (1.3e-5 with the interface image
    only, the default -1.5e-5; the base switched its pairs at six segment
    lengths in one step, the blend of D2 reaches 7.5).
    """
    houses = _houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0))
    base = _BASE["houses_500_100_h5"]
    assert abs(_z(houses, "h00", image_far_field="point") / base - 1.0) < 1e-6
    assert abs(_z(houses, "h00") / base - 1.0) > 2e-5
    rod = _rod_through()
    base = _BASE["rod_through_500_100"]
    assert abs(_z(rod, "r", image_far_field="point") / base - 1.0) < 2e-5


def test_point_leaves_out_the_far_work(monkeypatch) -> None:
    """No far-image evaluation; the near image pairs stay Galerkin integrals.

    On the cross-layer path every image of the families keeps its Galerkin
    integral within 7.5 segment lengths, the segment's own images included
    (audit W3R2A-01): 16 house grids below a 0.5 m top layer integrate
    208 768 image pairs in point form and 254 848 in the default form, whose
    self images reach to the end of the taper of the far term. Up to that
    audit only the interface image of pairs in one layer was integrated
    (128 832 pairs), as in the base of wave 3 (W3A-07), and long segments
    below the interface lost up to 58 %.
    """
    from tests.test_two_layer_far_images import _image_work

    world = _houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5))
    quadrupole = _image_work(world, monkeypatch)
    point = _image_work(world, monkeypatch, image_far_field="point")
    assert quadrupole["far"] > 0 and point["far"] == 0
    assert 0 < point["galerkin_short"] <= quadrupole["galerkin_short"]


def test_the_option_round_trips_and_is_kept_by_copies() -> None:
    """JSON, ``with_frequencies``, ``model_copy``, the provenance block and
    ``create_engine`` keep the option; the result records it."""
    eng = gf.Engine(image_far_field="point")
    assert (
        gf.Engine.model_validate_json(eng.model_dump_json()).image_far_field == "point"
    )
    assert eng.with_frequencies(50.0, 1000.0).image_far_field == "point"
    assert eng.model_copy(update={"segment_length": 1.0}).image_far_field == "point"
    assert gf.create_engine(image_far_field="point").image_far_field == "point"
    with pytest.raises(ValueError):
        gf.Engine(image_far_field="dipole")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eng.solve(_strip())
    assert res.metadata["series"]["far_field"] == "point"
    rebuilt = gf.Engine.model_validate(res.metadata["provenance"]["engine"])
    assert rebuilt.image_far_field == "point"


def test_the_field_and_the_mutual_paths_use_the_form_of_the_solve() -> None:
    """``FieldResult.potential`` and ``solve_mutual_field`` follow the option.

    The surface potential above the strip of 10.28 differs between the two
    forms by about 0.9 % (the point form of the far images, amplified by the
    alternating series of a conductive lower layer); each form's potential
    equals its mutual field.
    """
    pts = np.array([[5.0, 1.0, 0.0], [5.0, 3.0, 0.0], [12.0, -1.0, 0.0]])
    phi, mutual = {}, {}
    for form in ("quadrupole", "point"):
        eng = gf.Engine(segment_length=0.5, image_far_field=form)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eng.solve(_strip())
            phi[form] = res.potential(pts).real
            mutual[form] = np.asarray(gf.solve_mutual_field(_strip(), eng, ["s"], pts))
        current = float(np.real(res.electrode_currents["s"][0]))
        z = float(res.cluster_impedance("s")[0].real)
        # 1 A into the strip: the mutual field is the potential per ampere.
        np.testing.assert_allclose(
            mutual[form][:, 0].real, phi[form] / current, rtol=1e-9
        )
        assert z > 0.0
    assert np.max(np.abs(phi["point"] / phi["quadrupole"] - 1.0)) > 1e-3


def test_compare_engines_and_sweep_carry_the_option() -> None:
    """The two forms are two computations for ``compare_engines`` (no alias
    note) and a sweep axis of an engine factory."""
    from groundfield.postprocess.sweep import sweep
    from groundfield.validation import compare_engines

    world = _houses(gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = compare_engines(
            world,
            {
                "quadrupole": gf.Engine(segment_length=0.5),
                "point": gf.Engine(segment_length=0.5, image_far_field="point"),
            },
            rel_tolerance=1e-3,
        )
    assert report.is_consistent
    assert not any("same computation" in note for note in report.notes)
    deviations = [row.deviation for row in report.rows]
    assert max(deviations) > 1e-5  # the forms differ by 1.8e-4 here

    def factory(form: str) -> gf.Engine:
        return gf.Engine(segment_length=0.5, image_far_field=form)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sweep(
            lambda form: world,
            factory,
            axes={"form": ["quadrupole", "point"]},
        )
    values = dict(zip(table["form"], table["Z_re"], strict=True))
    assert abs(values["point"] / values["quadrupole"] - 1.0) > 1e-5


# audit W3R2A-02
@pytest.mark.parametrize("bad", ["Point", "quadrupol", "dipole", "", None])
def test_w3r2a_02_solve_image_2layer_rejects_an_unknown_far_field(bad) -> None:
    """``groundfield.solver.solve_image_2layer`` is public; its ``far_field``
    was not validated: every value other than ``"quadrupole"`` ran the point
    form, the bad value was recorded in ``metadata["series"]["far_field"]``,
    and ``FieldResult.potential`` then evaluated the quadrupole form
    (``series_far_field`` falls back), so the solve and its field disagreed
    (auditor W3R2A: ``"Point"`` was accepted and solved as ``"point"``).
    ``None`` takes the engine's value."""
    from groundfield.solver import solve_image_2layer

    eng = gf.Engine(segment_length=0.5, frequencies=[50.0])
    if bad is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = solve_image_2layer(_strip(), eng, far_field=bad)
        assert res.metadata["series"]["far_field"] == "quadrupole"
        return
    with pytest.raises(ValueError, match="far_field"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solve_image_2layer(_strip(), eng, far_field=bad)


# audit W3R2A-02
@pytest.mark.parametrize("bad", ["Point", "", 1])
def test_w3r2a_02_the_kernels_reject_an_unknown_far_field(bad) -> None:
    """The kernel constructors of the image family check the form too."""
    from groundfield.solver._kernels import TwoLayerSeriesKernel, _line_field_correction
    from groundfield.solver._line_kernel import FAR_FIELD_FORMS
    from groundfield.solver._series import ImageSeries
    from groundfield.solver.image_2layer import _LayeredImageFamilies

    assert FAR_FIELD_FORMS == ("quadrupole", "point")
    soil = gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5)
    series = ImageSeries.for_soil(soil, 300, 1e-6)
    line = _line_segments()
    with pytest.raises(ValueError, match="far_field"):
        TwoLayerSeriesKernel(500.0, series, far_field=bad)
    with pytest.raises(ValueError, match="far_field"):
        TwoLayerSeriesKernel(500.0, series, line=line, far_field=bad)
    with pytest.raises(ValueError, match="far_field"):
        _LayeredImageFamilies(line, line.mid[:, 2] >= 0.5, soil, bad)
    with pytest.raises(ValueError, match="far_field"):
        _line_field_correction(
            np.zeros((1, 3)), line, [np.ones(2)], 1e-3, None, far_field=bad
        )
    for good in FAR_FIELD_FORMS:
        TwoLayerSeriesKernel(500.0, series, line=line, far_field=good)
        _LayeredImageFamilies(line, line.mid[:, 2] >= 0.5, soil, good)


def _line_segments():
    """Two horizontal 0.5 m segments, one above and one below h_1 = 0.5 m."""
    from groundfield.solver._line_kernel import LineSegments

    mid = np.array([[0.0, 0.0, 0.3], [5.0, 0.0, 0.8]])
    return LineSegments(
        mid, np.full(2, 0.5), np.tile([1.0, 0.0, 0.0], (2, 1)), np.full(2, 0.005)
    )


#: Exact uniform-current resistance in Ohm of auditor W3R2A's 75 m strip
#: (a = 5 mm) at 0.8 m, wholly below a 0.5 m top layer: ``thinwire_ref.mutual``
#: (layer-2 images from the reflection coefficient, surface image with the
#: reduced kernel, ``n_x = 1200``; ``r9_point_long_segments.jsonl``).
_STRIP_BELOW = {(500.0, 100.0): 2.8838260805736358, (1000.0, 30.0): 0.8784333704746048}


def _strip_below(rho1, rho2) -> gf.World:
    w = gf.create_world(soil=gf.TwoLayerSoil(rho_1=rho1, rho_2=rho2, h_1=0.5))
    gf.create_electrode(
        w, "strip", name="s", start=(0, 0, 0.8), end=(75, 0, 0.8), wire_radius=0.005
    )
    gf.create_source(w, attached_to="s", magnitude=1.0)
    return w


# audit W3R2A-01
@pytest.mark.parametrize("ds", [1.0, 2.5, 7.5, 25.0, 37.5])
@pytest.mark.parametrize(
    "soil", [(500.0, 100.0), (1000.0, 30.0)], ids=["500_100", "1000_30"]
)
def test_w3r2a_01_the_point_form_keeps_long_segments_below_the_interface(
    soil, ds
) -> None:
    """The strip is one current owner: its resistance does not depend on ds.

    Both forms are within 1e-5 of the exact uniform-current value at every
    ds (the default within 1.3e-6). Before, ``image_far_field="point"`` kept
    only the interface image of the cross-layer correction in line form, so
    the segment's own surface and transmitted images, 1.6 m and more away,
    were point sources: -1.0e-3 at 2.5 m, -2.7e-2 at 7.5 m, -0.22 at 25 m,
    -0.37 at 37.5 m in 500/100 Ohm m (-0.58 in 1000/30; auditor W3R2A, R9).
    """
    rho1, rho2 = soil
    exact = _STRIP_BELOW[soil]
    z = {}
    for form in ("quadrupole", "point"):
        z[form] = _z_ds(_strip_below(rho1, rho2), "s", ds, image_far_field=form)
    assert abs(z["quadrupole"] / exact - 1.0) < 2e-6, z
    assert abs(z["point"] / exact - 1.0) < 1e-5, z


def _z_ds(world, anchor, ds, **kw) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = gf.Engine(segment_length=ds, frequencies=[50.0], **kw).solve(world)
    return float(res.cluster_impedance(anchor)[0].real)


# audit W3R2A-01
def test_w3r2a_01_the_series_path_of_the_point_form_is_unchanged() -> None:
    """The fix is on the cross-layer path: within the upper layer the point
    form keeps its value bit for bit (16 house grids in 1000/30 Ohm m with
    h_1 = 1.5 m: 19.047706773 Ohm at f4ad499 with and without numba, the
    default 19.044207405 Ohm)."""
    world = _houses(gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=1.5))
    point = _z(world, "h00", image_far_field="point")
    assert point == pytest.approx(19.047706772961927, rel=1e-12)


# audit W3R2A-01
def test_w3r2a_01_a_network_with_long_pieces_below_the_interface() -> None:
    """16 house grids below a 0.5 m top layer (500/100 Ohm m) and a bare
    galvanic 75 m conductor in two 37.5 m pieces: the point form within 2e-5
    of the default (before: +1.13 %, auditor W3R2A, R7)."""
    world = _houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5))
    gf.create_conductor(
        world,
        name="bare",
        start="h30",
        end="h33",
        conductor_type="bare_copper",
        cross_section=50e-6,
        coupling_to_soil="galvanic",
        discretize_segment_length=40.0,
    )
    quadrupole = _z(world, "h00")
    point = _z(world, "h00", image_far_field="point")
    assert abs(point / quadrupole - 1.0) < 2e-5, (point, quadrupole)


# audit W3R2A-04
def test_w3r2a_04_the_point_form_moves_house_grids_near_the_interface_by_4e_4() -> None:
    """16 house grids 0.4 m above a 1000/10 Ohm m interface (h_1 = 1.2 m): the
    point form reads +4.25e-4 against the default (auditor W3R2A, R2). The
    texts said 2e-5 … 1.8e-4 for networks; they now give 4.3e-4 as the upper
    end (``tests/test_docs_line_kernel_texts.py``)."""
    world = _houses(gf.TwoLayerSoil(rho_1=1000.0, rho_2=10.0, h_1=1.2))
    quadrupole = _z(world, "h00")
    point = _z(world, "h00", image_far_field="point")
    assert 1.8e-4 < point / quadrupole - 1.0 <= 4.3e-4, (point, quadrupole)
