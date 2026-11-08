"""Opt-in tabulated assembly of the Sommerfeld correction (finding 10.60).

``build_sommerfeld_correction_matrices(..., method="tabulated")`` shares one
table of the spectral integral S(rho) per pair of depth groups and frequency
and evaluates the outer quadrature of all segment pairs vectorised. These
tests pin its agreement with the exact per-pair assembly and its wiring into
the solver (an engine attribute ``sommerfeld_assembly``).
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import groundfield as gf
from groundfield.conductors import Conductor
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    build_sommerfeld_correction_matrices,
    build_sommerfeld_correction_matrix,
)

pytestmark = pytest.mark.finding("10.60")

OMEGAS = [2.0 * math.pi * 50.0, 2.0 * math.pi * 1000.0]
EARTHS = {
    "homogeneous": LayeredEarth(rhos=(100.0,), thicknesses=()),
    "two_layer": LayeredEarth(rhos=(500.0, 100.0), thicknesses=(5.0,)),
}


def _street(n: int = 12, *, drops: bool = True) -> np.ndarray:
    """A 5 m-segment street at 0.6 m with inclined 0.6 -> 0.8 m drops."""
    eps = [[[5.0 * k, 0.0, 0.6], [5.0 * (k + 1), 0.0, 0.6]] for k in range(n)]
    if drops:
        for x in (7.5, 22.5, 37.5):
            for k in range(5):
                eps.append(
                    [
                        [x, 0.8 * k, 0.6 + 0.04 * k],
                        [x, 0.8 * (k + 1), 0.6 + 0.04 * (k + 1)],
                    ]
                )
    return np.array(eps, dtype=float)


def _deviation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("earth", list(EARTHS), ids=list(EARTHS))
def test_tabulated_matches_the_exact_assembly(earth: str) -> None:
    """Street plus inclined drops (6 depth groups, every block tabulated)."""
    eps = _street()
    kw = dict(omegas=OMEGAS, earth=EARTHS[earth], image="none")
    exact = build_sommerfeld_correction_matrices(eps, **kw)
    tab = build_sommerfeld_correction_matrices(eps, method="tabulated", **kw)
    assert _deviation(tab, exact) < 3e-5
    np.testing.assert_allclose(tab, np.swapaxes(tab, 1, 2), rtol=0, atol=1e-15)


def test_tabulated_handles_segments_split_at_the_surface_and_an_interface() -> None:
    """Small blocks (a riser through z = 0, a rod through h1) use the
    per-pair rule; the result stays close to the exact assembly."""
    eps = [[[5.0 * k, 0.0, 0.6], [5.0 * (k + 1), 0.0, 0.6]] for k in range(10)]
    eps += [
        [[50.0, 0.0, 0.6], [50.0, 0.0, -2.0]],
        [[50.0, 0.0, -2.0], [50.0, 6.0, -2.0]],
    ]
    eps += [[[0.0, 1.0, 0.6], [0.0, 1.0, 7.0]]]
    eps = np.array(eps)
    kw = dict(omegas=OMEGAS, earth=EARTHS["two_layer"], image="none")
    exact = build_sommerfeld_correction_matrices(eps, **kw)
    tab = build_sommerfeld_correction_matrices(eps, method="tabulated", **kw)
    assert np.all(np.isfinite(tab))
    assert _deviation(tab, exact) < 3e-5


def test_tabulated_image_conventions_and_single_frequency() -> None:
    eps = _street(8, drops=False)
    earth = EARTHS["homogeneous"]
    many = build_sommerfeld_correction_matrices(
        eps, omegas=OMEGAS, earth=earth, image="additive", method="tabulated"
    )
    one = build_sommerfeld_correction_matrix(
        eps, omega=OMEGAS[1], earth=earth, image="additive", method="tabulated"
    )
    assert np.array_equal(many[1], one)
    zero = build_sommerfeld_correction_matrices(
        eps, omegas=[0.0], earth=earth, method="tabulated"
    )
    assert not zero.any()


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        build_sommerfeld_correction_matrices(
            _street(2, drops=False),
            omegas=OMEGAS,
            earth=EARTHS["homogeneous"],
            method="fast",
        )


def _pen_world() -> gf.World:
    w = gf.create_world(soil=gf.HomogeneousSoil(rho=100.0))
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5), length=3.0)
    gf.create_electrode(w, "rod", name="g2", position=(80.0, 0.0, 0.5), length=3.0)
    w.add_conductor(
        Conductor(
            name="pen",
            start=(0.0, 0.0, 0.7),
            end=(80.0, 0.0, 0.7),
            start_electrode="g1",
            end_electrode="g2",
            cross_section=95e-6,
            discretize_segment_length=5.0,
            inductance_model="neumann",
        )
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_branch_assembly_reads_the_engine_option() -> None:
    """``assemble_branches`` passes ``engine.sommerfeld_assembly`` on."""
    from groundfield.solver import _pipeline

    w = _pen_world()
    eng = gf.create_engine(
        backend="image", frequencies=[50.0, 1000.0], sommerfeld_assembly="exact"
    )
    disc = _pipeline.discretise(w, eng)
    topo = _pipeline.build_topology(w, disc)
    exact = _pipeline.assemble_branches(w, eng, topo).earth_correction
    duck = SimpleNamespace(
        earth_inductive_model="sommerfeld",
        frequencies=[50.0, 1000.0],
        sommerfeld_assembly="tabulated",
    )
    tab = _pipeline.assemble_branches(w, duck, topo).earth_correction
    om = 2.0 * math.pi * 1000.0
    assert not np.array_equal(tab(om), exact(om))
    assert _deviation(tab(om), exact(om)) < 3e-5


def test_solve_with_an_engine_that_selects_the_tabulated_assembly() -> None:
    """End to end through ``Engine.sommerfeld_assembly``."""
    kw = dict(backend="image", segment_length=0.5, frequencies=[50.0, 1000.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_exact = gf.Engine(**kw, sommerfeld_assembly="exact").solve(_pen_world())
        res_tab = gf.Engine(**kw, sommerfeld_assembly="tabulated").solve(_pen_world())
    assert (
        res_tab.metadata["provenance"]["engine"]["sommerfeld_assembly"] == "tabulated"
    )
    for name in ("g1", "g2"):
        z_exact = np.asarray(res_exact.electrode_potentials[name])
        z_tab = np.asarray(res_tab.electrode_potentials[name])
        assert not np.array_equal(z_tab, z_exact)
        np.testing.assert_allclose(z_tab, z_exact, rtol=1e-6)


def test_the_assembly_is_a_validated_engine_field() -> None:
    """``Engine.sommerfeld_assembly``: default, validation, JSON (coup2 request).

    The default is ``"tabulated"`` since the user decision Q16 (2026-11-08).
    """
    assert gf.Engine().sommerfeld_assembly == "tabulated"
    assert gf.create_engine(sommerfeld_assembly="tabulated").sommerfeld_assembly == (
        "tabulated"
    )
    with pytest.raises(ValueError, match="tabulated"):
        gf.Engine(sommerfeld_assembly="table")
    engine = gf.Engine(sommerfeld_assembly="tabulated")
    assert gf.Engine.model_validate_json(engine.model_dump_json()) == engine
    # engine JSON written before the field existed loads with the default
    old = {
        k: v for k, v in gf.Engine().model_dump().items() if k != "sommerfeld_assembly"
    }
    assert gf.Engine.model_validate(old).sommerfeld_assembly == "tabulated"


@pytest.mark.slow
def test_conductors_in_the_surface_plane_fall_back_to_the_per_pair_rule() -> None:
    """A block whose lambda grid cannot resolve its largest distance (a line
    at z = 0: decay length 0) keeps the per-pair rule instead of warning."""
    eps = np.array([[[5.0 * k, 0.0, 0.0], [5.0 * (k + 1), 0.0, 0.0]] for k in range(9)])
    kw = dict(omegas=OMEGAS[1:], earth=EARTHS["homogeneous"], image="none")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tab = build_sommerfeld_correction_matrices(eps, method="tabulated", **kw)
    exact = build_sommerfeld_correction_matrices(eps, **kw)
    np.testing.assert_allclose(tab, exact, rtol=1e-12, atol=0.0)
