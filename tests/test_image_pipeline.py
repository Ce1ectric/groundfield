"""Shared image-family pipeline and Tagg/Sunde series (review pass 10, 10.75).

``solve_image`` and ``solve_image_2layer`` used to carry two copies of
the solve flow, and the two-layer image series existed in five copies
with two truncation policies. Both backends now run through
``solver._pipeline.solve_image_family`` and every consumer of the
series iterates one :class:`~groundfield.solver._series.ImageSeries`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver import _pipeline
from groundfield.solver._series import ImageSeries, _series_tail_bound


def _rod_world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="rod", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w, "ring", name="ring", center=(6.0, 0.0, 0.8), radius=2.0)
    gf.create_source(w, attached_to="rod", magnitude=1.0)
    return w


@pytest.mark.finding("10.75")
@pytest.mark.parametrize(
    "soil",
    [
        gf.HomogeneousSoil(resistivity=100.0),
        gf.TwoLayerSoil(rho_1=300.0, rho_2=60.0, h_1=4.0),
    ],
    ids=["homogeneous", "two_layer"],
)
def test_image_backends_share_one_pipeline(soil, monkeypatch) -> None:
    """Both entry points run the same discretise → … → result steps."""
    calls: list[str] = []
    original = _pipeline.solve

    def spy(*args, **kwargs):
        calls.append("solve")
        return original(*args, **kwargs)

    monkeypatch.setattr(_pipeline, "solve", spy)
    res = _rod_world(soil).solve(gf.Engine(segment_length=0.5))
    assert calls == ["solve"]
    assert res.backend == (
        "image" if "Homogeneous" in type(soil).__name__ else "image_2layer"
    )


@pytest.mark.finding("10.75")
def test_image_series_is_shared_by_solver_field_and_mutual(monkeypatch) -> None:
    """Solver, ``FieldResult.potential`` and ``solve_mutual_*`` iterate one series."""
    users: list[int] = []
    original = ImageSeries.iter_layer_terms

    def spy(self):
        users.append(self.max_terms)
        return original(self)

    monkeypatch.setattr(ImageSeries, "iter_layer_terms", spy)
    soil = gf.TwoLayerSoil(rho_1=300.0, rho_2=60.0, h_1=4.0)
    w = _rod_world(soil)
    eng = gf.Engine(segment_length=0.5, image_max_terms=300)
    res = w.solve(eng)
    assert users and set(users) == {300}
    users.clear()
    res.potential(np.array([[3.0, 0.0, 0.0]]))
    assert users and set(users) == {100}
    users.clear()
    w2 = gf.create_world(soil=soil)
    gf.create_electrode(w2, "rod", name="a", position=(0.0, 0.0, 0.5), length=2.0)
    gf.create_electrode(w2, "rod", name="b", position=(9.0, 0.0, 0.5), length=2.0)
    gf.solve_mutual_matrix(w2, eng, ["a", "b"], np.array([[0.0, 0, 0], [9.0, 0, 0]]))
    assert users


@pytest.mark.finding("10.75")
@pytest.mark.parametrize("K", [-0.94, -0.5, 0.0, 0.3, 0.818])
def test_image_series_truncation_and_report(K: float) -> None:
    series = ImageSeries.from_coefficients(K, 5.0, max_terms=300, tol=1e-6)
    n = series.n_terms
    assert 1 <= n <= 300
    assert series.converged == (_series_tail_bound(abs(K), n) < 1e-6)
    if n > 1:  # the previous term had not reached the tolerance yet
        assert _series_tail_bound(abs(K), n - 1) >= 1e-6
    offsets, signs, weights = series.offsets()
    assert offsets.shape == signs.shape == weights.shape == (2 + 4 * n,)
    np.testing.assert_array_equal(weights[:2], [1.0, 1.0])
    np.testing.assert_allclose(weights[2::4], K ** np.arange(1, n + 1))
    rep = series.report()
    assert rep["n_terms_used"] == n and rep["converged"] == series.converged


@pytest.mark.finding("10.75")
def test_two_layer_metadata_comes_from_the_series() -> None:
    soil = gf.TwoLayerSoil(rho_1=1000.0, rho_2=30.0, h_1=5.0)
    eng = gf.Engine(segment_length=1.0, image_max_terms=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _rod_world(soil).solve(eng)
    series = ImageSeries.for_soil(soil, 20, eng.image_series_tol)
    assert res.metadata["n_terms_used"] == series.n_terms == 20
    assert res.metadata["converged"] is series.converged is False
