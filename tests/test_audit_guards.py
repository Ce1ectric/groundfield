"""Regression tests for the WP-A guard patch (audit 2026-07-08).

Covers the silent-wrong-answer class found by the physics/numerics
audit:

* A1 — surface-laid (z = 0) conductor with Neumann inductance used to
  produce a degenerate self-image inductance (~4.7 H instead of
  ~55 µH for a 25 m segment) that silently blocked the branch
  current. Now a hard ``ValueError``.
* A2 — a distributed conductor whose ``discretize_segment_length``
  exceeds its length (``n_segments == 1``) used to vanish from the
  nodal system entirely (open circuit). Now it falls back to a
  lumped finite branch.
* A3 — ``segment_length <= wire_radius`` used to flip the sign of the
  thin-wire self-term and solve to nonsense. Now a hard
  ``ValueError`` from the discretiser; ``Engine.solve`` additionally
  emits the advisory ``check_segment_resolution`` findings as
  warnings.
* A4 — series truncation: tail-bound criterion
  :math:`|K|^{n+1}/(1-|K|) < tol`, a visible
  ``SeriesTruncationWarning``, and the engine knobs
  ``image_max_terms`` / ``image_series_tol`` reaching mom/cim/bem.
* A5 — concrete-shell worlds (ADR-0012) are rejected loudly by the
  backends that do not implement the shell correction.
* A6 — cross-layer geometries: cim/bem/mom_sommerfeld n = 2 now
  dispatch the rigorous cross-layer kernel instead of silently
  applying the upper-layer series; n >= 3 raises.
* A7 — non-current sources warn instead of being dropped silently;
  overlapping segments trip the clamp warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.inductance import build_inductance_matrix
from groundfield.solver.image_2layer import SeriesTruncationWarning

SOIL = gf.HomogeneousSoil(resistivity=100.0)


def _two_rod_world(*, z_head: float = 0.7, **conductor_kwargs):
    """Two 3-m rods 100 m apart, connected by one conductor."""
    w = gf.create_world(soil=SOIL)
    g1 = gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, z_head),
                             length=3.0, wire_radius=0.005)
    g2 = gf.create_electrode(w, "rod", name="g2",
                             position=(100.0, 0.0, z_head),
                             length=3.0, wire_radius=0.005)
    gf.create_conductor(w, name="pen", start=g1, end=g2,
                        cross_section=50e-6, **conductor_kwargs)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


# ---------------------------------------------------------------------
# A1 — degenerate mirror geometry for surface-laid inductive conductors
# ---------------------------------------------------------------------


def test_surface_segment_with_image_raises() -> None:
    """A z = 0 segment coincides with its own mirror image."""
    eps = np.array([[[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]]])
    with pytest.raises(ValueError, match="soil-surface plane"):
        build_inductance_matrix(eps, np.array([0.005]), use_image=True)


def test_surface_segment_without_image_is_fine() -> None:
    eps = np.array([[[0.0, 0.0, 0.0], [25.0, 0.0, 0.0]]])
    L = build_inductance_matrix(eps, np.array([0.005]), use_image=False)
    # Free-space thin-wire self-inductance of 25 m / 5 mm: ~42 µH.
    assert 3e-5 < L[0, 0] < 6e-5


def test_buried_segment_with_image_is_sane() -> None:
    eps = np.array([[[0.0, 0.0, 0.7], [25.0, 0.0, 0.7]]])
    L = build_inductance_matrix(eps, np.array([0.005]), use_image=True)
    # Additive image raises the value above free space, but it must
    # stay in the µH range (the degenerate case gave ~4.7 H).
    assert L[0, 0] < 1e-4


def test_surface_pen_with_neumann_raises_via_solve() -> None:
    """The end-to-end trap: rods with heads at z = 0, inductive PEN."""
    w = _two_rod_world(z_head=0.0, discretize_segment_length=25.0,
                       inductance_model="neumann")
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.raises(ValueError, match="soil-surface plane"):
        eng.solve(w)


# ---------------------------------------------------------------------
# A2 — n_segments == 1 distributed conductor must not vanish
# ---------------------------------------------------------------------


def test_single_segment_distributed_conductor_conducts() -> None:
    """dsl >= conductor length used to open-circuit the conductor."""
    w = _two_rod_world(discretize_segment_length=100.0)  # n_segments == 1
    eng = gf.create_engine(backend="image", segment_length=0.5)
    res = eng.solve(w)
    i2 = res.electrode_currents["g2"][0]
    # Correct current split between two equal rods is ~0.5 each
    # (series resistance 0.034 Ohm << rod resistance ~35 Ohm).
    assert abs(i2) == pytest.approx(0.5, abs=0.05)


def test_single_segment_distributed_matches_lumped() -> None:
    w_dist = _two_rod_world(discretize_segment_length=100.0)
    w_lump = _two_rod_world()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    z_dist = eng.solve(w_dist).cluster_impedance("g1")[0]
    z_lump = eng.solve(w_lump).cluster_impedance("g1")[0]
    assert z_dist == pytest.approx(z_lump, rel=1e-12)


# ---------------------------------------------------------------------
# A3 — thin-wire guard
# ---------------------------------------------------------------------


def test_segment_length_below_wire_radius_raises() -> None:
    """Small ring, thick wire: segment 0.039 m < a = 0.05 m."""
    w = gf.create_world(soil=SOIL)
    gf.create_electrode(w, "ring", name="g1", center=(0.0, 0.0, 0.7),
                        radius=0.05, wire_radius=0.05)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.raises(ValueError, match="wire_radius"):
        eng.solve(w)


def test_engine_solve_emits_resolution_findings_as_warnings() -> None:
    """Advisory findings (thin-wire ratio < 5) surface as warnings."""
    w = gf.create_world(soil=SOIL)
    # 1 m rod with a = 0.2 m: segments 0.5 m -> ratio 2.5 (< 5) but
    # still > a, so it solves — with a warning.
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0),
                        length=1.0, wire_radius=0.2)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.warns(UserWarning, match="thin-wire|resolution"):
        eng.solve(w)


# ---------------------------------------------------------------------
# A4 — truncation warning + engine knob forwarding
# ---------------------------------------------------------------------

HARD_SOIL = gf.TwoLayerSoil(rho_1=30.0, rho_2=1000.0, h_1=5.0)  # K = +0.94


def _rod_world(soil):
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0),
                        length=3.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_truncation_emits_warning() -> None:
    """max_terms far below the tail requirement must warn visibly."""
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    eng = eng.model_copy(update={"image_max_terms": 20})
    with pytest.warns(SeriesTruncationWarning):
        res = eng.solve(_rod_world(HARD_SOIL))
    assert res.metadata["converged"] is False


def test_default_max_terms_converges_at_k094() -> None:
    """The raised default (250) covers the AP1 corner |K| = 0.94."""
    eng = gf.create_engine(backend="image_2layer", segment_length=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SeriesTruncationWarning)
        res = eng.solve(_rod_world(HARD_SOIL))
    assert res.metadata["converged"] is True


def test_engine_knob_reaches_mom() -> None:
    eng = gf.create_engine(backend="mom", segment_length=0.5)
    eng = eng.model_copy(update={"image_max_terms": 20})
    with pytest.warns(SeriesTruncationWarning):
        res = eng.solve(_rod_world(HARD_SOIL))
    assert res.metadata["converged"] is False


@pytest.mark.parametrize("backend", ["mom", "cim", "bem"])
def test_layered_backends_agree_at_high_contrast(backend: str) -> None:
    """With the knob forwarded, all n = 2 backends still agree with
    image_2layer at the AP1 corner contrast."""
    eng_ref = gf.create_engine(backend="image_2layer", segment_length=0.5)
    z_ref = eng_ref.solve(_rod_world(HARD_SOIL)).cluster_impedance("g1")[0]
    eng = gf.create_engine(backend=backend, segment_length=0.5)
    z = eng.solve(_rod_world(HARD_SOIL)).cluster_impedance("g1")[0]
    assert z.real == pytest.approx(z_ref.real, rel=0.05)


# ---------------------------------------------------------------------
# A5 — concrete shells rejected by non-image backends
# ---------------------------------------------------------------------


def _shelled_world():
    w = gf.create_world(soil=SOIL)
    gf.create_electrode(
        w, "strip", name="g1",
        start=(0.0, 0.0, 0.7), end=(10.0, 0.0, 0.7),
        wire_radius=0.005,
        concrete_shell_coefficient_ohm_m=100.0,
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


@pytest.mark.parametrize("backend", ["mom", "cim", "bem"])
def test_concrete_shell_rejected_by_non_image_backends(backend: str) -> None:
    eng = gf.create_engine(backend=backend, segment_length=0.5)
    with pytest.raises(NotImplementedError, match="concrete-shell"):
        eng.solve(_shelled_world())


def test_concrete_shell_still_works_on_image() -> None:
    eng = gf.create_engine(backend="image", segment_length=0.5)
    res = eng.solve(_shelled_world())
    assert res.cluster_impedance("g1")[0].real > 0.0


# ---------------------------------------------------------------------
# A6 — cross-layer consistency for the n = 2 backends
# ---------------------------------------------------------------------


def _cross_layer_world():
    """Rod from 0.5 m to 7.5 m through the interface at h1 = 5 m."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=300.0, h_1=5.0)
    w = gf.create_world(soil=soil)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.5),
                        length=7.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_cross_layer_n2_cim_matches_image_2layer() -> None:
    """cim used to apply the upper-layer series silently; it now
    dispatches the same ADR-0007 cross-layer kernel as image_2layer.
    Both use the average-potential scheme, so they must agree
    tightly."""
    eng_ref = gf.create_engine(backend="image_2layer", segment_length=1.0)
    z_ref = eng_ref.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    eng = gf.create_engine(backend="cim", segment_length=1.0)
    z = eng.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    assert z.real == pytest.approx(z_ref.real, rel=0.05)


def test_cross_layer_n2_bem_consistent_with_image_2layer() -> None:
    """bem now dispatches the ADR-0007 cross-layer kernel too, but it
    *solves* the per-segment current distribution instead of
    prescribing uniform leakage. Across a layer interface the
    uniform-leakage assumption of the image family carries a
    noticeable bias (audit 2026-07-08, P5), so only a loose
    consistency bracket is asserted here; the solved distribution is
    expected *below* the uniform-leakage value for rho_2 > rho_1."""
    eng_ref = gf.create_engine(backend="image_2layer", segment_length=1.0)
    z_ref = eng_ref.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    eng = gf.create_engine(backend="bem", segment_length=1.0)
    z = eng.solve(_cross_layer_world()).cluster_impedance("g1")[0]
    assert 0.7 * z_ref.real < z.real <= 1.05 * z_ref.real


# ---------------------------------------------------------------------
# A7 — source and clamp diagnostics
# ---------------------------------------------------------------------


def test_overlapping_electrodes_warn() -> None:
    """Two coincident rods trip the clamp warning during assembly."""
    w = gf.create_world(soil=SOIL)
    gf.create_electrode(w, "rod", name="g1", position=(0.0, 0.0, 0.0),
                        length=3.0, wire_radius=0.005)
    gf.create_electrode(w, "rod", name="g2", position=(0.0, 0.0, 0.0),
                        length=3.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    eng = gf.create_engine(backend="image", segment_length=0.5)
    with pytest.warns(UserWarning, match="clamp"):
        eng.solve(w)
