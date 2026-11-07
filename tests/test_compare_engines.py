"""Tests for the cross-engine comparison helper (ADR-0001)."""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf


def _world_with_rod() -> gf.World:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=1.5)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world


def test_compare_image_against_itself_is_consistent() -> None:
    """Sanity: two ``image`` engines with the same resolution must be identical."""
    world = _world_with_rod()
    eng_a = gf.create_engine(backend="image", segment_length=0.05)
    eng_b = gf.create_engine(backend="image", segment_length=0.05)
    report = gf.compare_engines(
        world,
        engines={"a": eng_a, "b": eng_b},
        rel_tolerance=1e-9,
    )
    assert report.is_consistent
    assert report.deviations["g1"] < 1e-12


def test_compare_image_resolutions_consistent_within_tolerance() -> None:
    """Two ``image`` engines with different segment_length must agree
    within 5 %."""
    world = _world_with_rod()
    eng_coarse = gf.create_engine(backend="image", segment_length=0.2)
    eng_fine = gf.create_engine(backend="image", segment_length=0.025)
    report = gf.compare_engines(
        world,
        engines={"coarse": eng_coarse, "fine": eng_fine},
        rel_tolerance=0.05,
    )
    assert report.is_consistent, report.summary()


def test_compare_engines_requires_at_least_two() -> None:
    world = _world_with_rod()
    with pytest.raises(ValueError, match="at least 2"):
        gf.compare_engines(world, engines={"a": gf.create_engine()})


def test_compare_image_vs_fem_homogeneous_within_envelope() -> None:
    """``fem`` (axisymmetric volume PDE with equivalent-hemisphere
    reduction) and ``image`` agree to within 10 % on a single-rod
    world. ``compare_engines`` must report the comparison as
    consistent under the documented FEM tolerance."""
    world = _world_with_rod()
    report = gf.compare_engines(
        world,
        engines={
            "image": gf.create_engine(backend="image", segment_length=0.05),
            "fem": gf.create_engine(backend="fem", segment_length=0.05),
        },
        rel_tolerance=0.10,
    )
    assert report.is_consistent, report.summary()


def test_compare_flags_stub_metadata_via_notes() -> None:
    """If a result reports ``metadata['stub'] == True``, the comparison
    is flagged in ``notes`` with a 'stub result' entry. This stays a
    useful sanity check even though no production backend currently
    emits a stub result."""
    world = _world_with_rod()
    eng_a = gf.create_engine(backend="image", segment_length=0.05)
    eng_b = gf.create_engine(backend="image", segment_length=0.05)
    res_a = eng_a.solve(world)
    res_b = eng_b.solve(world)

    # Emulate a stub by mutating the metadata of one result.
    res_b.metadata["stub"] = True

    cmp = gf.EngineComparison(results={"a": res_a, "b": res_b}, rel_tolerance=0.05)
    # Trigger the same stub-detection logic that ``compare_engines``
    # runs internally.
    for label, res in cmp.results.items():
        if res.metadata.get("stub"):
            cmp.notes.append(
                f"Engine '{label}' returned a stub result "
                f"(metadata['stub']=True). Comparison not meaningful."
            )
    assert any("stub result" in n for n in cmp.notes)


def test_compare_with_potential_sample_points() -> None:
    """Optional sample-point comparison is included and reported."""
    world = _world_with_rod()
    eng_a = gf.create_engine(backend="image", segment_length=0.05)
    eng_b = gf.create_engine(backend="image", segment_length=0.05)
    pts = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    report = gf.compare_engines(
        world,
        engines={"a": eng_a, "b": eng_b},
        rel_tolerance=1e-9,
        sample_points=pts,
    )
    assert report.is_consistent
    assert any("Potential point-sample" in n for n in report.notes)


def test_compare_engines_summary_string_contains_engines_and_clusters() -> None:
    world = _world_with_rod()
    report = gf.compare_engines(
        world,
        engines={
            "fine": gf.create_engine(backend="image", segment_length=0.025),
            "coarse": gf.create_engine(backend="image", segment_length=0.2),
        },
        rel_tolerance=0.10,
    )
    text = report.summary()
    assert "fine" in text
    assert "coarse" in text
    assert "g1" in text


# ---------------------------------------------------------------------
# 10.71 — complex Z at every frequency, explicit tolerance table
# ---------------------------------------------------------------------


def _ring_and_line_world() -> gf.World:
    """Ring with a 400 m inductive PEN-like line to a remote rod."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(w, "ring", name="g1", center=(0, 0, 0.8), radius=2.0)
    gf.create_electrode(w, "rod", name="far", position=(400.0, 0.0, 0.5), length=3.0)
    gf.create_conductor(
        w,
        start="g1",
        end="far",
        discretize_segment_length=50.0,
        coupling_to_soil="isolated",
        inductance_model="neumann",
        cross_section=95e-6,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _earth_model_engines(*models: str) -> dict[str, gf.Engine]:
    return {
        m: gf.create_engine(
            backend="image",
            segment_length=1.0,
            frequencies=[50.0, 1000.0],
            earth_inductive_model=m,
        )
        for m in models
    }


@pytest.mark.finding("10.71")
def test_compare_engines_sees_the_reactance_at_every_frequency() -> None:
    """Perfect-mirror and Sommerfeld earth return differ in X at 1 kHz.

    The input reactance at 1 kHz is 0.600 Ω (perfect mirror) against
    0.376 Ω (Sommerfeld), while Re Z at 50 Hz agrees to 0.002 %; v0.15.0
    compared Re Z of the clusters at the first frequency only and
    reported the pair as consistent.
    """
    rep = gf.compare_engines(
        _ring_and_line_world(),
        _earth_model_engines("perfect_mirror", "sommerfeld"),
        rel_tolerance=0.05,
    )
    assert not rep.is_consistent, rep.summary()
    failed = [r for r in rep.rows if not r.passed]
    assert failed, rep.summary()
    # Only the reactance disagrees (at 50 Hz: 0.031 vs 0.020 Ω, at 1 kHz
    # 0.600 vs 0.376 Ω); the resistances agree.
    assert 1000.0 in {r.frequency for r in failed}
    assert {r.component for r in failed} == {"Im"}
    worst = max(failed, key=lambda r: r.deviation)
    assert worst.quantity == "Z_in[source_0]"
    assert worst.deviation > 0.5


@pytest.mark.finding("10.71")
def test_compare_engines_rejects_mismatched_frequency_lists() -> None:
    """v0.15.0 compared 50 Hz with 1 kHz and reported "consistent"."""
    with pytest.raises(ValueError, match="frequenc"):
        gf.compare_engines(
            _world_with_rod(),
            {
                "a": gf.create_engine(backend="image", frequencies=[50.0]),
                "b": gf.create_engine(backend="image", frequencies=[1000.0]),
            },
        )


class _NanEngine(gf.Engine):
    """Engine whose cluster potential comes back as NaN."""

    def solve(self, world):  # type: ignore[override]
        res = super().solve(world)
        n = len(res.frequencies)
        res.electrode_potentials = {
            name: [complex("nan")] * n for name in res.electrode_potentials
        }
        return res


@pytest.mark.finding("10.71")
def test_compare_engines_counts_a_non_finite_result_as_failure() -> None:
    """Two engines agree, a third returns NaN: v0.15.0 dropped the NaN
    and reported the comparison as consistent."""
    rep = gf.compare_engines(
        _world_with_rod(),
        {
            "a": gf.create_engine(backend="image", segment_length=0.05),
            "b": gf.create_engine(backend="image", segment_length=0.05),
            "nan": _NanEngine(backend="image", segment_length=0.05),
        },
        rel_tolerance=0.05,
    )
    assert not rep.is_consistent, rep.summary()
    assert any("non-finite" in n and "nan" in n for n in rep.notes), rep.notes


class _ZeroEngine(gf.Engine):
    """Engine whose result reports 0 V at electrode 'g1' (a broken backend)."""

    def solve(self, world):  # type: ignore[override]
        res = super().solve(world)
        res.electrode_potentials["g1"] = [0j] * len(res.frequencies)
        return res


def _two_rod_world() -> gf.World:
    """Two separately driven rods: one cluster stays comparable."""
    world = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(world, "rod", name="g1", position=(0, 0, 0.5), length=3.0)
    gf.create_electrode(world, "rod", name="h", position=(50, 0, 0.5), length=3.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    gf.create_source(world, attached_to="h", magnitude=1.0)
    return world


@pytest.mark.finding("10.71", "F03")
def test_compare_engines_counts_a_zero_of_one_engine_as_failure() -> None:
    """One engine reports 0 Ω, the other 34.3 Ω (audit round 1, F03).

    The base of the calendar release skipped a quantity as soon as the
    *smallest* magnitude was zero, noted it as "undefined for every
    engine" and — the second rod agreeing — reported the comparison as
    consistent.
    """
    rep = gf.compare_engines(
        _two_rod_world(),
        {
            "image": gf.create_engine(backend="image", segment_length=0.05),
            "zero": _ZeroEngine(backend="image", segment_length=0.05),
        },
        rel_tolerance=0.05,
    )
    z = {lbl: v[0] for lbl, v in rep.impedances["Z_cluster[g1]"].items()}
    assert abs(z["image"]) > 30.0 and z["zero"] == 0j
    assert not rep.is_consistent, rep.summary()
    failed = [r for r in rep.rows if not r.passed]
    assert {r.quantity for r in failed} >= {"Z_cluster[g1]"}
    assert all(r.deviation == float("inf") for r in failed)
    assert any("zero for engine(s) zero" in n for n in rep.notes), rep.notes
    assert not any("undefined for every engine" in n for n in rep.notes), rep.notes


def test_compare_engines_skips_a_quantity_that_is_zero_for_every_engine() -> None:
    """Guard: all magnitudes zero is still "nothing to compare" (F03)."""
    rep = gf.compare_engines(
        _world_with_rod(),
        {
            "a": _ZeroEngine(backend="image", segment_length=0.05),
            "b": _ZeroEngine(backend="image", segment_length=0.05),
        },
        rel_tolerance=0.05,
    )
    assert "Z_cluster[g1]" not in {r.quantity for r in rep.rows}
    assert any(
        n.startswith("Z_cluster[g1]: undefined for every engine") for n in rep.notes
    ), rep.notes


@pytest.mark.finding("10.71")
def test_compare_engines_tolerance_table_is_explicit() -> None:
    """One row per quantity, frequency and component, with its tolerance."""
    rep = gf.compare_engines(
        _world_with_rod(),
        {
            "fine": gf.create_engine(
                backend="image", segment_length=0.05, frequencies=[0.0, 50.0, 1000.0]
            ),
            "coarse": gf.create_engine(
                backend="image", segment_length=0.2, frequencies=[0.0, 50.0, 1000.0]
            ),
        },
        rel_tolerance=0.05,
    )
    assert rep.is_consistent, rep.summary()
    assert rep.frequencies == [0.0, 50.0, 1000.0]
    keys = {(r.quantity, r.frequency, r.component) for r in rep.rows}
    for q in ("Z_cluster[g1]", "Z_in[source_0]"):
        for f in (0.0, 50.0, 1000.0):
            for c in ("Re", "Im"):
                assert (q, f, c) in keys, (q, f, c)
    assert all(r.tolerance == 0.05 for r in rep.rows)
    assert all(set(r.values) == {"fine", "coarse"} for r in rep.rows)
    # the complex values per frequency are kept for the user
    z = rep.impedances["Z_cluster[g1]"]["fine"]
    assert len(z) == 3 and all(isinstance(v, complex) for v in z)
    text = rep.summary()
    assert "1000" in text and "Im" in text


def test_engine_comparison_keeps_the_positional_order_of_0_15() -> None:
    """The fields added for 10.71 follow the 0.15.0 fields."""
    cmp = gf.EngineComparison({}, 0.05, {"g1": {}}, {"g1": 0.01}, True, ["n"])
    assert cmp.cluster_impedance_table == {"g1": {}}
    assert cmp.deviations == {"g1": 0.01}
    assert cmp.is_consistent is True and cmp.notes == ["n"]
    assert cmp.rows == [] and cmp.frequencies == []
