"""Regression tests for the Tier-1 performance path (ADR-0010, WP-F).

Covers the three mechanisms introduced by the 2026-07-08 audit WP-F:

* batched multiport-Z assembly (one kernel call with an
  ``(n_segments, N_a)`` excitation matrix),
* the ``multiport_cache`` reuse across per-frequency calls,
* equivalence of the batched path with the historic per-column
  construction.
"""

from __future__ import annotations

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver.image import (
    _self_corrected_kernel,
    _solve_cluster_currents,
)

SOIL = gf.HomogeneousSoil(resistivity=100.0)
RHO = 100.0


def _three_rod_setup():
    """Three rods, one source, one finite conductor — small but with
    every structural element of the augmented system."""
    w = gf.create_world(soil=SOIL)
    g1 = gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.0),
                             length=3.0, wire_radius=0.005)
    g2 = gf.create_electrode(w, "rod", name="g2", position=(10.0, 0, 0.0),
                             length=3.0, wire_radius=0.005)
    gf.create_electrode(w, "rod", name="g3", position=(20.0, 0, 0.0),
                        length=3.0, wire_radius=0.005)
    gf.create_conductor(w, name="c12", start=g1, end=g2,
                        cross_section=50e-6)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def test_batched_z_equals_per_column_loop() -> None:
    """The one-shot (N, N_a) assembly must reproduce the historic
    per-column construction to FP accuracy."""
    w = _three_rod_setup()
    eng = gf.create_engine(backend="image", segment_length=0.5)
    # Build the discretisation exactly as solve_image does.
    from groundfield.solver.image import _discretize_electrode

    segs = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in w.electrodes:
        s = _discretize_electrode(e, eng.segment_length)
        elec_to_segidx[e.name] = list(range(len(segs), len(segs) + len(s)))
        segs.extend(s)
    pts = np.array([s.midpoint for s in segs])
    lens = np.array([s.length for s in segs])
    radii = np.array([s.wire_radius for s in segs])
    names = [e.name for e in w.electrodes]
    n = pts.shape[0]

    def kernel(p, ls, r, c):
        return _self_corrected_kernel(p, ls, r, c, RHO)

    # Historic per-column loop.
    Z_ref = np.zeros((len(names), len(names)))
    for j, nj in enumerate(names):
        idxs_j = elec_to_segidx[nj]
        tc = np.zeros(n)
        tc[idxs_j] = lens[idxs_j] / lens[idxs_j].sum()
        phi = kernel(pts, lens, radii, tc)
        for i, ni in enumerate(names):
            Z_ref[i, j] = float(phi[elec_to_segidx[ni]].mean())

    # Batched path.
    exc = np.zeros((n, len(names)))
    for j, nj in enumerate(names):
        idxs_j = elec_to_segidx[nj]
        exc[idxs_j, j] = lens[idxs_j] / lens[idxs_j].sum()
    phi_all = kernel(pts, lens, radii, exc)
    Z_new = np.empty_like(Z_ref)
    for i, ni in enumerate(names):
        Z_new[i, :] = phi_all[elec_to_segidx[ni], :].mean(axis=0)

    np.testing.assert_allclose(Z_new, Z_ref, rtol=1e-13, atol=0.0)


def test_multiport_cache_is_filled_and_reused() -> None:
    """The second call with the same cache must not invoke the kernel."""
    w = _three_rod_setup()
    from groundfield.solver.image import (
        _build_clusters,
        _build_finite_branches,
        _discretize_electrode,
    )

    segs = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in w.electrodes:
        s = _discretize_electrode(e, 0.5)
        elec_to_segidx[e.name] = list(range(len(segs), len(segs) + len(s)))
        segs.extend(s)
    pts = np.array([s.midpoint for s in segs])
    lens = np.array([s.length for s in segs])
    radii = np.array([s.wire_radius for s in segs])
    cluster_id = _build_clusters(w.electrodes, w.conductors)
    branches = _build_finite_branches(w.conductors, cluster_id)

    calls = {"n": 0}

    def counting_kernel(p, ls, r, c):
        calls["n"] += 1
        return _self_corrected_kernel(p, ls, r, c, RHO)

    common = dict(
        electrodes=w.electrodes,
        elec_input_current={"g1": 1.0 + 0j, "g2": 0j, "g3": 0j},
        cluster_id=cluster_id,
        seg_points=pts,
        seg_lengths=lens,
        wire_radii=radii,
        elec_to_segidx=elec_to_segidx,
        self_kernel=counting_kernel,
        finite_branches=branches,
    )
    cache: dict = {}
    r1 = _solve_cluster_currents(**common, multiport_cache=cache)
    assert calls["n"] == 1
    assert "Z" in cache
    r2 = _solve_cluster_currents(**common, multiport_cache=cache)
    assert calls["n"] == 1  # kernel NOT called again
    for k in r1:
        assert r1[k] == pytest.approx(r2[k], rel=0.0, abs=0.0)


def test_multifrequency_inductive_solve_consistent() -> None:
    """Multi-frequency inductive solve: the cached-Z path must give a
    frequency-monotone, finite response and reproduce the
    single-frequency solve exactly at the matching frequency."""
    def build():
        w = gf.create_world(soil=SOIL)
        g1 = gf.create_electrode(w, "rod", name="g1", position=(0, 0, 0.7),
                                 length=3.0, wire_radius=0.005)
        g2 = gf.create_electrode(w, "rod", name="g2",
                                 position=(100.0, 0, 0.7),
                                 length=3.0, wire_radius=0.005)
        gf.create_conductor(w, name="pen", start=g1, end=g2,
                            cross_section=50e-6,
                            discretize_segment_length=25.0,
                            inductance_model="neumann")
        gf.create_source(w, attached_to="g1", magnitude=1.0)
        return w

    eng_multi = gf.create_engine(
        backend="image", segment_length=0.5,
        frequencies=[50.0, 500.0, 1000.0],
    )
    res_multi = eng_multi.solve(build())
    eng_single = gf.create_engine(
        backend="image", segment_length=0.5, frequencies=[500.0],
    )
    res_single = eng_single.solve(build())
    i_multi = res_multi.electrode_currents["g2"][1]
    i_single = res_single.electrode_currents["g2"][0]
    assert i_multi == pytest.approx(i_single, rel=1e-12)
    # |I_g2| decreases with frequency (rising PEN reactance).
    mags = [abs(i) for i in res_multi.electrode_currents["g2"]]
    assert mags[0] > mags[1] > mags[2]


def test_multiport_z_exact_reciprocity_mixed_lengths() -> None:
    """Length-weighted row reduction restores exact Z symmetry
    (audit 2026-07-08, WP-B3).

    A mesh electrode with unequal longitudinal/transverse segment
    lengths plus a rod used to produce ``Z[i, j] != Z[j, i]`` at the
    discretisation level (length-weighted excitation columns paired
    with an *unweighted* potential average). With the Galerkin-
    consistent length-weighted average the multiport matrix is
    reciprocal to machine precision.
    """
    from groundfield.solver.image import (
        _build_clusters,
        _discretize_electrode,
    )

    w = gf.create_world(soil=SOIL)
    # 4.0 x 2.0 grid with a different pitch per axis (n_x=2 -> 2.0 m,
    # n_y=2 -> 1.0 m). With ds = 0.7 and crossing-aligned discretisation
    # the longitudinal wires get 0.667 m segments and the transverse
    # wires 0.5 m: unequal lengths inside one electrode.
    gf.create_electrode(w, "grid_mesh", name="g1", corner=(0.0, 0.0, 0.7),
                        size=(4.0, 2.0), n_x=2, n_y=2, wire_radius=0.005)
    gf.create_electrode(w, "rod", name="g2", position=(10.0, 0.0, 0.0),
                        length=3.0, wire_radius=0.005)
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    # A finite conductor keeps both electrodes in the active set.
    gf.create_conductor(w, name="c", start="g1", end="g2",
                        cross_section=50e-6)

    segs = []
    elec_to_segidx: dict[str, list[int]] = {}
    for e in w.electrodes:
        s = _discretize_electrode(e, 0.7)
        elec_to_segidx[e.name] = list(range(len(segs), len(segs) + len(s)))
        segs.extend(s)
    pts = np.array([s.midpoint for s in segs])
    lens = np.array([s.length for s in segs])
    assert np.unique(np.round(lens[elec_to_segidx["g1"]], 12)).size > 1
    radii = np.array([s.wire_radius for s in segs])
    cluster_id = _build_clusters(w.electrodes, w.conductors)
    from groundfield.solver.image import _build_finite_branches

    branches = _build_finite_branches(w.conductors, cluster_id)

    cache: dict = {}
    _solve_cluster_currents(
        electrodes=w.electrodes,
        elec_input_current={"g1": 1.0 + 0j, "g2": 0j},
        cluster_id=cluster_id,
        seg_points=pts,
        seg_lengths=lens,
        wire_radii=radii,
        elec_to_segidx=elec_to_segidx,
        self_kernel=lambda p, ls, r, c: _self_corrected_kernel(
            p, ls, r, c, RHO,
        ),
        finite_branches=branches,
        multiport_cache=cache,
    )
    Z = cache["Z"]
    asym = np.max(np.abs(Z - Z.T)) / np.max(np.abs(Z))
    assert asym < 1e-13
