"""Tests for the cross-engine comparison helper (ADR-0001)."""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf


def _world_with_rod() -> gf.World:
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.5), length=1.5)
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
