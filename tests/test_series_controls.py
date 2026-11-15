"""Series controls of the two-layer image series after the solve.

Review pass 10:

* 10.25 — :meth:`FieldResult.potential` and ``solve_mutual_*`` capped the
  Tagg/Sunde series at 100 terms whatever ``Engine.image_max_terms`` said
  (+0.83 % at 100 m for :math:`K = -0.94`), silently. They now use the
  controls recorded with the solve (``metadata["series"]``) and warn on
  truncation.
* 10.50 (F43) — cross-layer solves reported the term count and the
  truncation of a series they never evaluated.
* 10.74 — the backend entry points carried their own series defaults
  (100 / 200) instead of the engine's.

The reference values come from an independent numpy evaluation of the
series on the solved point sources (5000 terms), not from the package.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.image_2layer import SeriesTruncationWarning

# K = (30 - 1000) / (30 + 1000) = -0.942: the AP1 corner soil.
_HARD = dict(rho_1=1000.0, rho_2=30.0, h_1=5.0)


def _ring_world(soil: gf.TwoLayerSoil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.8), radius=5.0)
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


def _reference_series(
    points: np.ndarray, res: gf.FieldResult, soil: gf.TwoLayerSoil, n_terms: int
) -> np.ndarray:
    """Independent Tagg/Sunde sum of the point sources of ``res`` (upper layer)."""
    src = np.array([ps.position for ps in res.point_sources])
    cur = np.array([ps.current[0] for ps in res.point_sources])
    K, h, rho = soil.reflection_coefficient, soil.h_1, soil.rho_1
    s2 = (points[:, None, 0] - src[None, :, 0]) ** 2 + (
        points[:, None, 1] - src[None, :, 1]
    ) ** 2
    z, zs = points[:, None, 2], src[None, :, 2]
    total = 1.0 / np.sqrt(s2 + (z - zs) ** 2) + 1.0 / np.sqrt(s2 + (z + zs) ** 2)
    for n in range(1, n_terms + 1):
        quad = sum(
            1.0 / np.sqrt(s2 + (z - (sg * 2.0 * n * h + tz * zs)) ** 2)
            for sg in (1.0, -1.0)
            for tz in (1.0, -1.0)
        )
        total = total + K**n * quad
    return rho / (4.0 * np.pi) * (total @ cur)


_FAR = np.array([[100.0, 0.0, 0.0], [0.0, 200.0, 0.0], [300.0, 300.0, 0.0]])


@pytest.mark.finding("10.25")
def test_potential_uses_the_series_controls_of_the_engine() -> None:
    soil = gf.TwoLayerSoil(**_HARD)
    engine = gf.create_engine(
        kernel="point", segment_length=1.0, image_max_terms=2000, image_series_tol=1e-10
    )
    res = _ring_world(soil).solve(engine)
    phi = res.potential(_FAR)
    ref = _reference_series(_FAR, res, soil, 5000)
    np.testing.assert_allclose(phi, ref, rtol=1e-5)


@pytest.mark.finding("10.25")
def test_potential_without_a_series_record_uses_the_engine_defaults() -> None:
    """Results written before 2026.11 carry no record: defaults, not 100 terms."""
    soil = gf.TwoLayerSoil(**_HARD)
    res = _ring_world(soil).solve(gf.create_engine(kernel="point", segment_length=1.0))
    res.metadata.pop("series", None)
    phi = res.potential(_FAR)
    ref = _reference_series(_FAR, res, soil, 5000)
    np.testing.assert_allclose(phi, ref, rtol=1e-5)


@pytest.mark.finding("10.25")
def test_potential_warns_when_its_series_is_truncated() -> None:
    soil = gf.TwoLayerSoil(**_HARD)
    engine = gf.create_engine(segment_length=1.0, image_max_terms=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SeriesTruncationWarning)
        res = _ring_world(soil).solve(engine)
    with pytest.warns(SeriesTruncationWarning, match="FieldResult.potential"):
        res.potential(_FAR)


@pytest.mark.finding("10.25")
def test_mutual_field_uses_the_series_controls_of_the_engine() -> None:
    soil = gf.TwoLayerSoil(**_HARD)
    engine = gf.create_engine(
        kernel="point", segment_length=1.0, image_max_terms=2000, image_series_tol=1e-10
    )
    res = _ring_world(soil).solve(engine)
    world = gf.create_world(soil=soil)
    gf.create_electrode(world, "ring", name="ring", center=(0.0, 0.0, 0.8), radius=5.0)
    R = gf.solve_mutual_field(world, engine, ["ring"], _FAR)
    ref = _reference_series(_FAR, res, soil, 5000)
    np.testing.assert_allclose(R[:, 0], ref, rtol=1e-5)


@pytest.mark.finding("10.25")
def test_mutual_matrix_explicit_series_controls_still_apply() -> None:
    """An explicit ``matrix_max_terms`` keeps overriding the engine (and warns)."""
    soil = gf.TwoLayerSoil(**_HARD)
    world = gf.create_world(soil=soil)
    for name, x in (("a", 0.0), ("b", 40.0)):
        gf.create_electrode(world, "rod", name=name, position=(x, 0.0, 0.5), length=2.0)
    engine = gf.create_engine(segment_length=1.0)
    probes = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    with pytest.warns(SeriesTruncationWarning, match="solve_mutual"):
        Z_cut = gf.solve_mutual_matrix(
            world, engine, ["a", "b"], probes, offdiag="probe", matrix_max_terms=10
        )
    Z = gf.solve_mutual_matrix(world, engine, ["a", "b"], probes, offdiag="probe")
    assert abs(Z_cut[0, 1] / Z[0, 1] - 1.0) > 1e-3


@pytest.mark.finding("10.50")
def test_cross_layer_solve_reports_no_series() -> None:
    """The ADR-0007 path never evaluates the image series (F43)."""
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=10.0, h_1=5.0)  # K = -0.98
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="r", position=(0.0, 0.0, 0.5), length=12.0)
    gf.create_source(w, attached_to="r", magnitude=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeriesTruncationWarning)
        res = w.solve(gf.create_engine(segment_length=1.0))
    assert res.metadata["series_used"] is False
    assert res.metadata["n_terms_used"] == 0
    assert res.metadata["converged"] is True


@pytest.mark.finding("10.50")
def test_in_layer_solve_still_reports_its_series() -> None:
    soil = gf.TwoLayerSoil(**_HARD)
    res = _ring_world(soil).solve(gf.create_engine(segment_length=1.0))
    assert res.metadata["series_used"] is True
    assert res.metadata["n_terms_used"] == 277
    assert res.metadata["converged"] is True
    assert res.metadata["series"] == {
        "max_terms": 300,
        "tol": 1e-6,
        "far_field": "quadrupole",
    }


@pytest.mark.finding("10.74")
@pytest.mark.parametrize("entry", ["image_2layer", "image_nlayer"])
def test_backend_entry_points_take_the_engine_series_controls(entry: str) -> None:
    """A direct backend call follows the engine, not a private default."""
    from groundfield.solver.image_2layer import solve_image_2layer
    from groundfield.solver.image_nlayer import solve_image_nlayer

    solve = {"image_2layer": solve_image_2layer, "image_nlayer": solve_image_nlayer}
    soil = gf.TwoLayerSoil(**_HARD)
    engine = gf.Engine(segment_length=1.0, image_max_terms=7)
    with pytest.warns(SeriesTruncationWarning):
        res = solve[entry](_ring_world(soil), engine)
    assert res.metadata["n_terms_used"] == 7
    assert res.metadata["converged"] is False
