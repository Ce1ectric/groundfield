"""Conventions of the coupling package (review pass 10).

* 10.46 — one internal-inductance convention (``include_internal``),
  consistent with the per-metre references at GMR = a e^{-1/4};
* 10.95 — English prose in ``coupling/layered_green.py``.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from groundfield.coupling import layered_green
from groundfield.coupling.inductance import (
    build_carson_correction_matrix,
    build_inductance_matrix,
    thin_wire_self_inductance,
)
from groundfield.references.earth_return import carson_self_impedance


@pytest.mark.finding("10.95")
def test_layered_green_prose_is_english() -> None:
    doc = layered_green._leggauss_cached.__doc__ or ""
    german = re.compile(r"\b(fuer|Knoten|Gewichte|gecacht|wird|ohne|daher|Rueckgabe)\b")
    assert not german.search(doc), doc


@pytest.mark.finding("10.46")
@pytest.mark.parametrize(
    "include_internal, gmr_factor", [(True, math.exp(-0.25)), (False, 1.0)]
)
def test_internal_inductance_convention_matches_references(
    include_internal, gmr_factor
) -> None:
    """X' of a long buried wire (Carson on its anti-parallel baseline):
    include_internal=True <-> GMR = a e^{-1/4}; False <-> GMR = a."""
    a, h, rho, f, length = 0.004, 0.7, 100.0, 1000.0, 1000.0
    n = 100
    x = np.linspace(0.0, length, n + 1)
    eps = np.array([[[x[k], 0, h], [x[k + 1], 0, h]] for k in range(n)])
    radii = np.full(n, a)
    om = 2 * math.pi * f
    L = build_inductance_matrix(
        eps, radii, image="antiparallel", include_internal=include_internal
    )
    dz = build_carson_correction_matrix(eps, radii, omega=om, sigma_earth=1.0 / rho)
    x_per_m = (1j * om * L.sum() + dz.sum()).imag / length
    ref = carson_self_impedance(rho, f, gmr=a * gmr_factor).imag
    assert x_per_m == pytest.approx(ref, rel=1e-3)


@pytest.mark.finding("10.46")
def test_include_internal_per_segment() -> None:
    eps = np.array([[[0, 0, 0.7], [10, 0, 0.7]], [[0, 2, 0.7], [10, 2, 0.7]]], float)
    radii = np.array([0.004, 0.006])
    L_mix = build_inductance_matrix(
        eps, radii, image="none", include_internal=[True, False]
    )
    assert L_mix[0, 0] == thin_wire_self_inductance(10.0, 0.004, include_internal=True)
    assert L_mix[1, 1] == thin_wire_self_inductance(10.0, 0.006, include_internal=False)
    with pytest.raises(ValueError, match="include_internal"):
        build_inductance_matrix(eps, radii, include_internal=[True])


@pytest.mark.finding("10.20")
def test_image_and_use_image_are_exclusive_and_validated() -> None:
    eps = np.array([[[0, 0, 0.7], [10, 0, 0.7]]], float)
    with pytest.raises(TypeError):
        build_inductance_matrix(eps, [0.005], use_image=True, image="additive")
    with pytest.raises(ValueError, match="image"):
        build_inductance_matrix(eps, [0.005], image="mirror")
    assert np.array_equal(
        build_inductance_matrix(eps, [0.005], use_image=True),
        build_inductance_matrix(eps, [0.005], image="additive"),
    )
    assert np.array_equal(
        build_inductance_matrix(eps, [0.005], use_image=False),
        build_inductance_matrix(eps, [0.005], image="none"),
    )
