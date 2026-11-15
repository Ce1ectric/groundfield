"""``Engine.image_far_field``: the form of the far images of the line kernel.

The author's decision of 2026-11-15 (stream fw3xl): the quadrupole far images
of wave 3 (10.28) stay the default; ``image_far_field="point"`` offers the
far-image treatment before wave 3 for catalogue sweeps where speed matters
more than the last 1e-4: far Tagg/Sunde images and far images of the
cross-layer correction in point form, the cross-layer correction in line form
for the interface image of pairs in one layer only (its Galerkin integral for
near pairs and on the diagonal), the near-pair reach of the base; the
continuous switch of user decision D2 and the transverse distance of image
pairs (audit W3A-02) are kept. The option travels through ``Engine`` and the
registry, JSON, ``compare_engines`` and ``sweep``, ``FieldResult.potential``
(``metadata["series"]["far_field"]``) and ``solve_mutual_*``.
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
    """The point form is the far-image treatment of 34ad269.

    16 house grids in 500/100 Ohm m, h_1 = 5 m: every image is at least
    8.4 m (> 7.5 segment lengths) from every segment, so the base used the
    point form for all of them: ``"point"`` +2.4e-7 against the base (the
    blended switch and the radius term of direct pairs), the default +3.6e-5
    (the quadrupole terms). A rod through the interface keeps the base's line
    form of the cross-layer correction, the interface image: within 1.4e-5
    (the base switched its pairs at six segment lengths in one step, the
    blend of D2 reaches 7.5).
    """
    houses = _houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=5.0))
    base = _BASE["houses_500_100_h5"]
    assert abs(_z(houses, "h00", image_far_field="point") / base - 1.0) < 1e-6
    assert abs(_z(houses, "h00") / base - 1.0) > 2e-5
    rod = _rod_through()
    base = _BASE["rod_through_500_100"]
    assert abs(_z(rod, "r", image_far_field="point") / base - 1.0) < 2e-5


def test_point_leaves_out_the_far_work(monkeypatch) -> None:
    """No far-image evaluation, and on the cross-layer path only the interface
    image of pairs in one layer as Galerkin integrals (W3A-07: the private
    flag of wave 3 still integrated every image within 7.5 segment lengths)."""
    from tests.test_two_layer_far_images import _image_work

    world = _houses(gf.TwoLayerSoil(rho_1=500.0, rho_2=100.0, h_1=0.5))
    quadrupole = _image_work(world, monkeypatch)
    point = _image_work(world, monkeypatch, image_far_field="point")
    assert quadrupole["far"] > 0 and point["far"] == 0
    assert point["galerkin_short"] < 0.6 * quadrupole["galerkin_short"]


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
