"""Regression tests for review pass 9 — engine / world / validation API.

Covers four findings of the 2026-07-28 adversarial review that all sit
on the orchestration layer rather than in a physics kernel:

F18
    ``Engine.solve`` forwarded ``image_max_terms`` / ``image_series_tol``
    to ``solve_image_2layer`` but called ``solve_image_nlayer`` without
    arguments, so the multilayer dispatch silently used the backend's
    own defaults (200, 1e-6). The image-charge series of the Tagg/Sunde
    2-layer kernel converges like the geometric tail
    ``|K|^(n+1) / (1 - |K|)``; for the AP1 corner soils (|K| → 1) the
    documented remedy is to raise ``Engine.image_max_terms``, which was
    a no-op on the ``MultiLayerSoil`` path.

F31
    ``World.solve`` restored its source snapshot by *rebinding* the
    list, which detached the objects returned by ``create_source`` from
    the world. The no-mutation contract must be kept while preserving
    object identity, so the canonical "mutate the handle, re-solve"
    sweep works.

F35
    ``compare_engines`` measured the deviation against the ensemble
    *mean*, which for two engines is exactly half the true pairwise
    disagreement, so a ``rel_tolerance`` gate admitted about twice the
    configured spread.

F36
    ``compare_engines`` reported ``is_consistent=True`` when no cluster
    could be evaluated at all (e.g. a world without a source), i.e. a
    green result from a comparison that validated nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.soil.models import HomogeneousSoil, MultiLayerSoil, SoilLayer
from groundfield.solver.engine import Engine
from groundfield.solver.image_2layer import SeriesTruncationWarning
from groundfield.solver.result import FieldResult
from groundfield.world import World

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

#: High-contrast 2-layer soil expressed as a ``MultiLayerSoil``:
#: rho_1 = 1000 Ohm*m over rho_2 = 10 Ohm*m at h_1 = 5 m gives
#: K = (10 - 1000) / (10 + 1000) = -0.980, so the image series needs
#: ~ log(tol * (1 - |K|)) / log(|K|) >> 200 terms.
_RHO_1, _RHO_2, _H_1 = 1000.0, 10.0, 5.0


def _rod_world(soil) -> World:
    """3 m vertical rod fed with 1 A, one cluster named ``g1``."""
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=3.0)
    gf.create_source(world, attached_to="g1", magnitude=1.0)
    return world


def _multilayer_world() -> World:
    return _rod_world(
        MultiLayerSoil(
            layers=[
                SoilLayer(resistivity=_RHO_1, thickness=_H_1),
                SoilLayer(resistivity=_RHO_2),
            ]
        )
    )


def _two_layer_world() -> World:
    return _rod_world(
        gf.TwoLayerSoil(rho_1=_RHO_1, rho_2=_RHO_2, h_1=_H_1)
    )


def _Z(result: FieldResult) -> float:
    return float(result.cluster_impedance("g1")[0].real)


# ----------------------------------------------------------------------
# F18 — series controls must reach the image_nlayer backend
# ----------------------------------------------------------------------


def test_image_nlayer_honours_engine_image_max_terms() -> None:
    """Raising ``image_max_terms`` must change the multilayer result.

    ``solve_image_nlayer`` re-dispatches to ``solve_image_2layer`` for
    n = 2, so the truncation order is directly observable in Z. Before
    the fix both engines silently used the backend default of 200 terms
    and returned bit-identical impedances.
    """
    engine_lo = gf.create_engine(
        backend="image", image_max_terms=50, image_series_tol=1e-9,
    )
    engine_hi = gf.create_engine(
        backend="image", image_max_terms=3000, image_series_tol=1e-9,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeriesTruncationWarning)
        z_lo = _Z(engine_lo.solve(_multilayer_world()))
        z_hi = _Z(engine_hi.solve(_multilayer_world()))

    assert z_lo != pytest.approx(z_hi, abs=1e-9), (
        "image_max_terms has no effect on the image_nlayer path: "
        f"50 terms -> {z_lo!r}, 3000 terms -> {z_hi!r}"
    )
    # The truncated series over-estimates Z for K < 0.
    assert z_lo > z_hi


def test_image_nlayer_converges_to_two_layer_reference() -> None:
    """With enough terms the ``MultiLayerSoil`` and ``TwoLayerSoil``
    spellings of the same soil must give the same impedance and no
    truncation warning.

    Both worlds are physically identical, so any difference is pure
    truncation error of the image series.
    """
    engine = gf.create_engine(
        backend="image", image_max_terms=3000, image_series_tol=1e-9,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z_multi = _Z(engine.solve(_multilayer_world()))
    truncation = [w for w in caught
                  if issubclass(w.category, SeriesTruncationWarning)]
    assert not truncation, (
        "image_nlayer still truncates despite image_max_terms=3000: "
        f"{[str(w.message) for w in truncation]}"
    )

    z_two = _Z(engine.solve(_two_layer_world()))
    assert z_multi == pytest.approx(z_two, rel=1e-12), (
        f"MultiLayerSoil {z_multi!r} != TwoLayerSoil {z_two!r}"
    )
    # Guard against a silent re-introduction of the 200-term default,
    # which returned 332.14413 instead of the converged 332.14270.
    assert z_multi == pytest.approx(332.142695, abs=1e-4)


def test_image_nlayer_series_tol_is_forwarded() -> None:
    """A loose ``image_series_tol`` must stop the series early.

    With the default cap the tolerance decides where the geometric tail
    bound is considered small enough; a very loose tolerance therefore
    truncates and must show up as a different Z.
    """
    engine_loose = gf.create_engine(
        backend="image", image_max_terms=3000, image_series_tol=1e-1,
    )
    engine_tight = gf.create_engine(
        backend="image", image_max_terms=3000, image_series_tol=1e-12,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeriesTruncationWarning)
        z_loose = _Z(engine_loose.solve(_multilayer_world()))
        z_tight = _Z(engine_tight.solve(_multilayer_world()))
    assert z_loose != pytest.approx(z_tight, abs=1e-9)


# ----------------------------------------------------------------------
# F31 — World.solve must not detach the caller's Source handles
# ----------------------------------------------------------------------


class _FieldMutatingEngine(Engine):
    """Engine that mutates source fields in flight (contract probe)."""

    def solve(self, world: World) -> FieldResult:  # type: ignore[override]
        world.sources[0].magnitude = 12345.0
        world.sources[0].return_to = "somewhere_else"
        return super().solve(world)


class _ListMutatingEngine(Engine):
    """Engine that rebinds ``world.sources`` and appends an entry."""

    def solve(self, world: World) -> FieldResult:  # type: ignore[override]
        world.sources = [s.model_copy(deep=True) for s in world.sources]
        extra = world.sources[0].model_copy(deep=True)
        extra.name = "injected_by_backend"
        world.sources.append(extra)
        return super().solve(world)


def test_world_solve_keeps_source_handle_attached() -> None:
    """``create_source`` handles stay live across ``World.solve``."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    src = world.sources[0]
    source_list = world.sources
    engine = gf.create_engine(backend="image")

    world.solve(engine)

    assert src is world.sources[0], (
        "World.solve detached the Source object returned by create_source"
    )
    assert source_list is world.sources, "World.solve rebound world.sources"


def test_world_solve_sees_mutated_source_handle() -> None:
    """Mutating the handle between two solves must change the result.

    Doubling the injected current doubles the electrode potential
    (the field problem is linear in the source current), so the second
    solve must return exactly twice the first one.
    """
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    src = world.sources[0]
    engine = gf.create_engine(backend="image")

    phi_1 = world.solve(engine).electrode_potentials["g1"][0]
    src.magnitude *= 2.0
    phi_2 = world.solve(engine).electrode_potentials["g1"][0]

    assert world.sources[0].magnitude == pytest.approx(2.0)
    assert abs(phi_2) == pytest.approx(2.0 * abs(phi_1), rel=1e-12), (
        f"source mutation ignored: {phi_1!r} -> {phi_2!r}"
    )


def test_world_solve_still_rolls_back_field_mutation() -> None:
    """The no-mutation contract survives the identity-preserving fix."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    src = world.sources[0]
    engine = _FieldMutatingEngine(backend="image")

    world.solve(engine)

    assert src is world.sources[0]
    assert src.magnitude == pytest.approx(1.0)
    assert src.return_to is None


def test_world_solve_rolls_back_list_mutation_preserving_identity() -> None:
    """A backend that rebinds / grows ``sources`` is rolled back."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    src = world.sources[0]
    source_list = world.sources
    engine = _ListMutatingEngine(backend="image")

    world.solve(engine)

    assert len(world.sources) == 1
    assert world.sources[0] is src
    assert world.sources is source_list


def test_world_solve_opt_out_leaves_mutation_visible() -> None:
    """``snapshot_sources=False`` keeps the documented opt-out."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    src = world.sources[0]
    engine = _FieldMutatingEngine(backend="image")

    world.solve(engine, snapshot_sources=False)

    assert src is world.sources[0]
    assert src.magnitude == pytest.approx(12345.0)


# ----------------------------------------------------------------------
# F35 — compare_engines metric must be pairwise, not mean-referenced
# ----------------------------------------------------------------------


class _ScaledSoilEngine(Engine):
    """Engine that solves the world with ``rho`` scaled by a factor.

    For a homogeneous soil and a fixed discretisation the grounding
    impedance is exactly proportional to the soil resistivity
    (Z = rho * f(geometry)), so this fabricates a *known* cross-engine
    disagreement of ``rho_scale - 1`` without touching any kernel.
    """

    rho_scale: float = 1.0

    def solve(self, world: World) -> FieldResult:  # type: ignore[override]
        scaled = world.model_copy(deep=True)
        scaled.soil = HomogeneousSoil(
            resistivity=world.soil.resistivity * self.rho_scale
        )
        return super().solve(scaled)


def _known_disagreement_report(
    rho_scale: float, rel_tolerance: float
) -> gf.EngineComparison:
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    return gf.compare_engines(
        world,
        engines={
            "ref": gf.create_engine(backend="image", segment_length=0.1),
            "scaled": _ScaledSoilEngine(
                backend="image", segment_length=0.1, rho_scale=rho_scale
            ),
        },
        rel_tolerance=rel_tolerance,
    )


def test_deviation_equals_true_pairwise_disagreement() -> None:
    """A fabricated 6 % disagreement must be reported as 6 %."""
    report = _known_disagreement_report(1.06, rel_tolerance=0.05)
    assert report.deviations["g1"] == pytest.approx(0.06, rel=1e-9), (
        "deviation is not the pairwise relative difference: "
        f"{report.deviations!r}"
    )


def test_six_percent_disagreement_fails_five_percent_gate() -> None:
    """The gate must trip where the user configured it.

    Pre-fix the mean-referenced metric reported
    0.03 / 1.03 = 2.91 % for this pair and signed the comparison off as
    consistent.
    """
    report = _known_disagreement_report(1.06, rel_tolerance=0.05)
    assert not report.is_consistent, report.summary()

    # ... and the metric is not over-strict either.
    ok = _known_disagreement_report(1.06, rel_tolerance=0.07)
    assert ok.is_consistent, ok.summary()


def test_deviation_is_about_twice_the_old_mean_referenced_value() -> None:
    """Pin the size of the change of definition for two engines.

    The old definition was ``max_k |Z_k - mean(Z)| / |mean(Z)|``; it is
    recomputed here from the reported impedance table so the change is
    visible in the test itself. For two values ``a < b`` the two metrics
    are related exactly by ``new / old = 1 + b / a``, i.e. a factor of
    almost exactly 2 in any regime where the engines nearly agree —
    which is precisely the regime the tolerance gate lives in.
    """
    report = _known_disagreement_report(1.06, rel_tolerance=0.05)
    zs = np.asarray(
        list(report.cluster_impedance_table["g1"].values()), dtype=float
    )
    old_metric = float(np.max(np.abs(zs - zs.mean())) / abs(zs.mean()))
    new_metric = report.deviations["g1"]
    assert new_metric == pytest.approx(
        old_metric * (1.0 + zs.max() / zs.min()), rel=1e-9
    )
    assert 2.0 <= new_metric / old_metric < 2.1
    # The whole point: the old number sat below the gate, the new one above.
    assert old_metric < 0.05 < new_metric


def test_identical_engines_still_report_zero_deviation() -> None:
    """The new metric keeps the trivial case exact."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    report = gf.compare_engines(
        world,
        engines={
            "a": gf.create_engine(backend="image", segment_length=0.05),
            "b": gf.create_engine(backend="image", segment_length=0.05),
        },
        rel_tolerance=1e-12,
    )
    assert report.deviations["g1"] == pytest.approx(0.0, abs=1e-15)
    assert report.is_consistent, report.summary()


def test_sample_point_note_states_the_metric() -> None:
    """The potential point-sample uses (and documents) the same metric."""
    world = _rod_world(HomogeneousSoil(resistivity=100.0))
    report = gf.compare_engines(
        world,
        engines={
            "a": gf.create_engine(backend="image", segment_length=0.05),
            "b": gf.create_engine(backend="image", segment_length=0.05),
        },
        rel_tolerance=1e-12,
        sample_points=np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    note = next(n for n in report.notes if "point-sample" in n)
    assert "max pairwise" in note
    assert report.is_consistent, report.summary()


# ----------------------------------------------------------------------
# F36 — a comparison without any evaluated cluster is not "consistent"
# ----------------------------------------------------------------------


def test_source_free_world_is_not_consistent() -> None:
    """Nothing comparable must never report a green result."""
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=3.0)

    report = gf.compare_engines(
        world,
        engines={
            "image": gf.create_engine(backend="image", segment_length=0.1),
            "mom": gf.create_engine(backend="mom", segment_length=0.1),
        },
        rel_tolerance=0.05,
    )

    assert report.deviations == {}
    assert not report.is_consistent, report.summary()
    assert any("vacuous" in n for n in report.notes), report.notes
    assert "INCONSISTENT" in report.summary()


def test_source_free_world_with_sample_points_is_not_consistent() -> None:
    """Zero-potential sample points do not rescue a vacuous comparison.

    Without a source every potential is identically zero, so the
    relative deviation is undefined at every sample point too — the
    report must stay red rather than counting "0 % deviation" as
    agreement.
    """
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=3.0)

    report = gf.compare_engines(
        world,
        engines={
            "image": gf.create_engine(backend="image", segment_length=0.1),
            "mom": gf.create_engine(backend="mom", segment_length=0.1),
        },
        rel_tolerance=0.05,
        sample_points=np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    assert not report.is_consistent, report.summary()
    assert any("excluded" in n for n in report.notes), report.notes
    assert any("vacuous" in n for n in report.notes), report.notes


def test_misspelled_attached_to_is_not_consistent() -> None:
    """The same guard catches a source that feeds nothing.

    ``Source.attached_to`` pointing at a name that does not exist leaves
    every cluster with sum(I) = 0, so no impedance is defined.
    """
    world = gf.create_world(soil=HomogeneousSoil(resistivity=100.0))
    gf.create_electrode(world, "rod", name="g1",
                        position=(0, 0, 0.0), length=3.0)
    world.sources.append(
        gf.CurrentSource(name="s_typo", attached_to="g_typo", magnitude=1.0)
    )

    report = gf.compare_engines(
        world,
        engines={
            "coarse": gf.create_engine(backend="image", segment_length=0.4),
            "fine": gf.create_engine(backend="image", segment_length=0.1),
        },
        rel_tolerance=0.05,
    )
    if report.deviations:
        pytest.skip("world does define a cluster impedance; guard not exercised")
    assert not report.is_consistent, report.summary()
