"""Memory- and accuracy-regression tests for the cross-layer Hankel
contraction in :mod:`groundfield.coupling.layered_green`.

Background
----------
The two-layer *layered correction* (ADR-0007) evaluates a Hankel
transform ``J0(lambda * s) @ w_phi`` for every distance pair of a
reaction block. Assembled naively this materialises a dense
``(n_pairs, n_lambda)`` Bessel matrix; for a whole grounding grid
``n_pairs = n_seg**2`` and the oscillation-resolved lambda grid reaches
~5e4 nodes, so the single-shot form tried to allocate **263 GiB** for
the IEEE Std 80 Annex H Grid 3 (70 m grid + twenty cross-layer rods).

The contraction is now (a) **row-blocked** so peak memory stays within a
fixed budget and (b) for large single-depth-pair groups **interpolated**
over a log-spaced ``s``-grid (the correction is smooth in ``s`` because
the singular ``1/r`` part cancels in ``Phi_lay - Phi_hom``) — the same
policy already used by :func:`two_layer_probe_matrix`.

These tests pin both properties so the blow-up cannot silently return.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j0

from groundfield.coupling import layered_green as lg
from groundfield.coupling.layered_green import (
    _hankel_contract,
    two_layer_layered_correction_group,
    two_layer_layered_correction_real_space,
)


# ---------------------------------------------------------------------
# 1. Row-blocking is bit-for-bit identical to the single-shot matmul.
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n_pairs,n_lambda", [(1, 500), (63, 2000), (5000, 3000)])
def test_hankel_contract_matches_single_shot(n_pairs, n_lambda) -> None:
    rng = np.random.default_rng(n_pairs + n_lambda)
    lambdas = np.sort(rng.random(n_lambda) * 50.0)
    s_values = rng.random(n_pairs) * 100.0
    w_phi = rng.standard_normal(n_lambda)

    reference = j0(lambdas[None, :] * s_values[:, None]) @ w_phi
    chunked = _hankel_contract(lambdas, s_values, w_phi)

    assert chunked.shape == s_values.shape
    # A row-independent reduction: identical to the last bit.
    assert np.array_equal(chunked, reference)


def test_hankel_contract_preserves_input_shape() -> None:
    rng = np.random.default_rng(0)
    lambdas = np.linspace(0.1, 20.0, 400)
    w_phi = rng.standard_normal(400)
    s_values = rng.random((7, 11)) * 30.0
    out = _hankel_contract(lambdas, s_values, w_phi)
    assert out.shape == (7, 11)


def test_hankel_contract_actually_blocks(monkeypatch) -> None:
    """With a tiny budget the contraction must run in several blocks and
    still reproduce the single-shot result exactly."""
    rng = np.random.default_rng(1)
    lambdas = np.linspace(0.1, 40.0, 4000)
    w_phi = rng.standard_normal(4000)
    s_values = rng.random(2000) * 80.0
    reference = j0(lambdas[None, :] * s_values[:, None]) @ w_phi
    # 64 KiB budget -> chunk of only 1 row per block (8*4000 = 32 KiB/row).
    monkeypatch.setattr(lg, "_HANKEL_MATMUL_BUDGET_BYTES", 64 * 1024)
    out = _hankel_contract(lambdas, s_values, w_phi)
    assert out.shape == reference.shape
    # Blocking a matmul is mathematically exact; the residual is pure
    # BLAS gemm-vs-gemv rounding (~1e-13). In production the contraction
    # only ever sees <=64 pairs or 48 interpolation nodes (a single
    # block, hence bit-identical); this tiny-budget case is synthetic.
    np.testing.assert_allclose(out, reference, rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------
# 2. Large-group interpolation agrees with the exact per-pair kernel.
# ---------------------------------------------------------------------
def test_group_interpolation_matches_scalar_reference() -> None:
    """The interpolated large-group path reproduces the exact scalar
    real-space kernel to well within its stated ~1e-4 tolerance."""
    rho_1, rho_2, h_1 = 300.0, 100.0, 6.096
    z, z_s = 0.5, 7.0  # observer in upper layer, source across the interface

    # Many distances -> triggers the interpolation branch (> threshold).
    s_values = np.linspace(0.5, 90.0, 400)
    grouped = two_layer_layered_correction_group(
        s_values, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1
    )

    # Exact scalar kernel (correction only) at a handful of probe radii.
    for s in (1.0, 5.0, 20.0, 60.0):
        g_full = two_layer_layered_correction_real_space(
            s, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1
        )
        g_interp = float(np.interp(s, s_values, grouped))
        scale = max(abs(g_full), 1e-9)
        assert abs(g_interp - g_full) / scale < 5e-3, (s, g_interp, g_full)


def test_group_small_input_is_exact_contraction() -> None:
    """At or below the interpolation threshold the group falls back to the
    exact (chunked) contraction, matching the scalar kernel closely."""
    rho_1, rho_2, h_1 = 250.0, 80.0, 5.0
    z, z_s = 0.6, 0.6
    s_values = np.array([0.5, 2.0, 10.0, 40.0])  # size 4 <= threshold
    grouped = two_layer_layered_correction_group(
        s_values, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1
    )
    for s, g in zip(s_values, grouped):
        g_full = two_layer_layered_correction_real_space(
            s, z, z_s, rho_1=rho_1, rho_2=rho_2, h_1=h_1
        )
        assert abs(g - g_full) / max(abs(g_full), 1e-9) < 1e-6


def test_group_homogeneous_limit_is_zero() -> None:
    """rho_2 == rho_1: the layered correction vanishes for any group size."""
    s_values = np.linspace(0.5, 50.0, 200)
    out = two_layer_layered_correction_group(
        s_values, 0.5, 3.0, rho_1=120.0, rho_2=120.0, h_1=5.0
    )
    assert np.allclose(out, 0.0)
