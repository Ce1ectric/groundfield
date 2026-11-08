"""The shallow-segment advisory follows the kernel (audit A1-numerics F08).

``ShallowSegmentWarning`` quantifies the bias of the *point* form of a
segment's self image (a point image at 2z instead of the image line) for
segments longer than four times their depth. With the exact line kernel
(``kernel="line"``, the default of the image family) the self image is a
Galerkin integral without that bias, so the warning's claim ("biased high
by N %") was false: a 10 m strip at 0.1 m depth gives the same impedance
with 2 m and with 0.25 m segments.
"""

from __future__ import annotations

import warnings

import pytest

import groundfield as gf
from groundfield.solver._guards import ShallowSegmentWarning


def _strip_world(soil=None) -> gf.World:
    w = gf.create_world(
        soil=soil if soil is not None else gf.HomogeneousSoil(rho=100.0)
    )
    gf.create_electrode(w, "strip", name="g", start=(0, 0, 0.1), end=(10, 0, 0.1))
    gf.create_source(w, attached_to="g", magnitude=1.0)
    return w


def _solve(engine: gf.Engine, soil=None) -> tuple[float, list[str]]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = _strip_world(soil).solve(engine)
    shallow = [
        str(r.message) for r in rec if issubclass(r.category, ShallowSegmentWarning)
    ]
    return res.cluster_impedance("g")[0].real, shallow


SOILS = {
    "homogeneous": None,
    "two_layer": gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0),
}


@pytest.mark.finding("F08")
@pytest.mark.parametrize("soil", list(SOILS))
@pytest.mark.parametrize("kernel", [None, "line"], ids=["default", "line"])
def test_the_line_kernel_is_not_warned_as_biased(kernel, soil) -> None:
    z_coarse, shallow = _solve(
        gf.Engine(segment_length=2.0, kernel=kernel), SOILS[soil]
    )
    z_fine, _ = _solve(gf.Engine(segment_length=0.25, kernel=kernel), SOILS[soil])
    assert not shallow, shallow[0]
    # No mesh bias to warn about: 2e-6 (homogeneous), 3e-5 (two-layer, whose
    # far Tagg/Sunde images stay in point form, Engine.kernel docstring),
    # against +25 % for the point kernel.
    assert z_coarse == pytest.approx(z_fine, rel=5e-4)


@pytest.mark.parametrize("soil", list(SOILS))
@pytest.mark.parametrize("backend", ["image", "mom"])
def test_the_point_form_still_warns_with_its_bias(backend, soil) -> None:
    """Guard: point kernel and the mom family keep the advisory."""
    kw = {"kernel": "point"} if backend == "image" else {}
    z_coarse, shallow = _solve(
        gf.Engine(backend=backend, segment_length=2.0, **kw), SOILS[soil]
    )
    z_fine, _ = _solve(
        gf.Engine(backend=backend, segment_length=0.25, **kw), SOILS[soil]
    )
    assert len(shallow) == 1
    assert "point kernel" in shallow[0] and "kernel='line'" in shallow[0]
    assert z_coarse > 1.1 * z_fine  # the bias the message quantifies


@pytest.mark.finding("F08")
def test_the_message_no_longer_claims_the_mom_family_is_silent() -> None:
    _, shallow = _solve(gf.Engine(backend="mom", segment_length=2.0))
    assert "do not emit this warning" not in shallow[0]
