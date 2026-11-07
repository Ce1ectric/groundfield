"""Which impedance the post-processing helpers report (10.18, 10.19).

Three impedances can be read from a solved world:

``"input"``
    Driving-point impedance (phi_attached - phi_return) / I_source — what
    an instrument or a fault sees at the feeding point
    (:meth:`FieldResult.input_impedance`).
``"cluster"``
    Cluster potential over the cluster's own soil leakage
    (:meth:`FieldResult.cluster_impedance`).
``"electrode"``
    Potential of one electrode over that electrode's own leakage
    (:meth:`FieldResult.grounding_impedance`).

As soon as conductors carry part of the injected current away (PEN,
shields, leads) the three differ. Up to v0.15.0 ``sweep`` reported the
cluster impedance and the rho-f exporters fitted the electrode impedance;
since review pass 10 every helper takes ``quantity=`` with ``"input"`` as
the default.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.postprocess.rho_f_standard import rho_f_standard_from_results
from groundfield.postprocess.vector_fitting import rho_f_from_field_result

FREQS = [0.0, 50.0, 1000.0]


def _chain_world(rho: float = 100.0, n_houses: int = 3) -> gf.World:
    """Station ring and house rods chained by finite-impedance PEN pieces
    (separate galvanic clusters, as in a TN network)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
    gf.create_electrode(w, "ring", name="station", center=(0.0, 0.0, 0.8), radius=2.0)
    prev = "station"
    for k in range(n_houses):
        name = f"house_{k}"
        gf.create_electrode(
            w, "rod", name=name, position=(40.0 * (k + 1), 0.0, 0.5), length=2.0
        )
        gf.create_conductor(
            w,
            name=f"pen_{k}",
            start=prev,
            end=name,
            conductor_type="pen",
            wire_radius=0.004,
            resistivity=2.82e-8,
            cross_section=50e-6,
            coupling_to_soil="isolated",
        )
        prev = name
    gf.create_source(w, attached_to="station", magnitude=1.0)
    return w


def _bonded_station(rho: float = 100.0) -> gf.World:
    """Station ring with two rods on ideal bonds: one galvanic cluster (R7-02)."""
    w = gf.create_world(soil=gf.HomogeneousSoil(resistivity=rho))
    gf.create_electrode(w, "ring", name="ring", center=(0.0, 0.0, 0.6), radius=4.0)
    gf.create_electrode(w, "rod", name="rod_a", position=(4.0, 0.0, 0.6), length=2.0)
    gf.create_electrode(w, "rod", name="rod_b", position=(-4.0, 0.0, 0.6), length=2.0)
    gf.create_conductor(w, name="b_a", start="ring", end="rod_a")
    gf.create_conductor(w, name="b_b", start="ring", end="rod_b")
    gf.create_source(w, attached_to="ring", magnitude=1.0)
    return w


def _engine() -> gf.Engine:
    return gf.create_engine(backend="image", segment_length=0.5, frequencies=FREQS)


def _solve(world: gf.World) -> gf.FieldResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return world.solve(_engine())


def _sweep(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gf.sweep(_chain_world, _engine(), axes={"rho": [100.0]}, **kw)


# ---------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------


@pytest.mark.finding("10.18")
def test_sweep_default_reports_the_input_impedance() -> None:
    """Default Z = EPR / I_source (R5-02: v0.15.0 reported the station's
    own leakage impedance, 13.1 Ohm against 7.1 Ohm for three houses)."""
    df = _sweep()
    res = _solve(_chain_world())
    z_in = np.asarray(res.input_impedance())
    np.testing.assert_allclose(df["Z_re"] + 1j * df["Z_im"], z_in, rtol=1e-12)
    # 1 A source: |Z| equals |EPR|.
    np.testing.assert_allclose(df["abs_Z"], df["abs_U_E"], rtol=1e-12)
    np.testing.assert_allclose(df["abs_I"], 1.0, rtol=1e-12)
    z_cl = np.asarray(res.cluster_impedance("station"))
    assert np.all(np.abs(z_cl) > 1.5 * np.abs(z_in))
    assert df.attrs["impedance_quantity"] == "input"


@pytest.mark.finding("10.18")
def test_sweep_cluster_quantity_reproduces_v0150() -> None:
    df = _sweep(quantity="cluster")
    res = _solve(_chain_world())
    np.testing.assert_allclose(
        df["Z_re"] + 1j * df["Z_im"], res.cluster_impedance("station"), rtol=1e-12
    )
    i_leak = np.asarray(res.electrode_currents["station"])
    np.testing.assert_allclose(df["I_re"] + 1j * df["I_im"], i_leak, rtol=1e-12)


@pytest.mark.finding("10.18")
def test_sweep_electrode_quantity_is_u_over_i_of_the_feeding_electrode() -> None:
    df = _sweep(quantity="electrode")
    res = _solve(_chain_world())
    np.testing.assert_allclose(
        df["Z_re"] + 1j * df["Z_im"], res.grounding_impedance("station"), rtol=1e-12
    )


@pytest.mark.finding("10.18")
def test_sweep_needs_a_source_name_with_several_current_sources() -> None:
    def two_sources(rho: float) -> gf.World:
        w = _chain_world(rho, n_houses=1)
        gf.create_source(w, attached_to="house_0", magnitude=0.5, name="second")
        return w

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="source="):
            gf.sweep(two_sources, _engine(), axes={"rho": [100.0]})
        df = gf.sweep(two_sources, _engine(), axes={"rho": [100.0]}, source="second")
    res = _solve(two_sources(100.0))
    np.testing.assert_allclose(
        df["Z_re"] + 1j * df["Z_im"], res.input_impedance("second"), rtol=1e-12
    )


@pytest.mark.finding("10.18")
def test_sweep_rejects_an_unknown_quantity() -> None:
    with pytest.raises(ValueError, match="quantity"):
        _sweep(quantity="leakage")


@pytest.mark.finding("10.18")
def test_convergence_study_defaults_to_the_input_impedance() -> None:
    w = _chain_world(n_houses=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = gf.convergence_study(w, _engine(), segment_lengths=[1.0, 0.5])
        res = w.solve(_engine())
    sub = df[df["segment_length_m"] == 0.5].sort_values("frequency_Hz")
    np.testing.assert_allclose(
        sub["Z_re"] + 1j * sub["Z_im"], res.input_impedance(), rtol=1e-12
    )
    assert df.attrs["impedance_quantity"] == "input"


# ---------------------------------------------------------------------
# rho-f exporters
# ---------------------------------------------------------------------


@pytest.mark.finding("10.19")
def test_rho_f_from_field_result_fits_the_input_impedance_by_default() -> None:
    res = _solve(_chain_world())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = rho_f_from_field_result(res, n_poles=1)
        fit_named = rho_f_from_field_result(res, "station", n_poles=1)
    np.testing.assert_allclose(fit.fit_values, res.input_impedance(), rtol=1e-12)
    np.testing.assert_allclose(fit_named.fit_values, fit.fit_values, rtol=0)


@pytest.mark.finding("10.19")
def test_rho_f_from_field_result_keeps_the_old_quantities_on_request() -> None:
    res = _solve(_chain_world())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_el = rho_f_from_field_result(res, "station", quantity="electrode", n_poles=1)
        f_cl = rho_f_from_field_result(res, "station", quantity="cluster", n_poles=1)
    np.testing.assert_allclose(f_el.fit_values, res.grounding_impedance("station"))
    np.testing.assert_allclose(f_cl.fit_values, res.cluster_impedance("station"))
    with pytest.raises(ValueError, match="electrode_name"):
        rho_f_from_field_result(res, quantity="cluster")


@pytest.mark.finding("10.19")
def test_input_quantity_refuses_an_electrode_that_is_not_fed() -> None:
    res = _solve(_chain_world())
    with pytest.raises(ValueError, match="quantity='electrode'"):
        rho_f_from_field_result(res, "house_1", n_poles=1)


@pytest.mark.finding("10.19")
def test_rho_f_standard_from_results_bonded_station() -> None:
    """R7-02: on a ring with two bonded rods the exporters fitted the ring's
    own U/I, +19 % above the cluster (= input) impedance."""
    rhos = [50.0, 100.0, 300.0]
    results = [_solve(_bonded_station(r)) for r in rhos]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = rho_f_standard_from_results(results, rhos, "ring")
        fit_el = rho_f_standard_from_results(
            results, rhos, "ring", quantity="electrode"
        )
    z_in = np.concatenate([r.input_impedance() for r in results])
    z_cl = np.concatenate([r.cluster_impedance("ring") for r in results])
    z_el = np.concatenate([r.grounding_impedance("ring") for r in results])
    np.testing.assert_allclose(fit.sample_Z, z_in, rtol=1e-12)
    np.testing.assert_allclose(z_in, z_cl, rtol=1e-12)  # single cluster, KCL
    np.testing.assert_allclose(fit_el.sample_Z, z_el, rtol=1e-12)
    assert np.all(z_el.real > 1.1 * z_in.real)


@pytest.mark.finding("10.19")
def test_rho_f_standard_from_results_tn_world_uses_the_input_impedance() -> None:
    rhos = [50.0, 200.0]
    results = [_solve(_chain_world(r)) for r in rhos]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = rho_f_standard_from_results(results, rhos)
        fit_cl = rho_f_standard_from_results(
            results, rhos, "station", quantity="cluster"
        )
    np.testing.assert_allclose(
        fit.sample_Z, np.concatenate([r.input_impedance() for r in results]), rtol=1e-12
    )
    np.testing.assert_allclose(
        fit_cl.sample_Z,
        np.concatenate([r.cluster_impedance("station") for r in results]),
        rtol=1e-12,
    )


@pytest.mark.finding("10.19")
def test_input_quantity_needs_the_source_snapshot() -> None:
    from groundfield.solver.result import FieldResult

    res = FieldResult(
        backend="image",
        frequencies=[0.0, 50.0],
        electrode_potentials={"g1": [2.0 + 0j, 2.0 + 0j]},
        electrode_currents={"g1": [1.0 + 0j, 1.0 + 0j]},
    )
    with pytest.raises(ValueError, match="quantity='electrode'"):
        rho_f_from_field_result(res, "g1", n_poles=1)
    fit = rho_f_from_field_result(res, "g1", quantity="electrode", n_poles=1)
    np.testing.assert_allclose(fit.fit_values, [2.0, 2.0])


# ---------------------------------------------------------------------
# current balance: split factor links the input and the cluster impedance
# ---------------------------------------------------------------------


def test_split_factor_links_input_and_cluster_impedance() -> None:
    """Z_in = s * Z_cluster for one source returning through remote earth."""
    w = _chain_world()
    res = _solve(w)
    for k in range(len(FREQS)):
        s = gf.split_factor(res, w, frequency_index=k)
        z_cl = res.cluster_impedance("station")[k]
        assert res.input_impedance()[k] == pytest.approx(s * z_cl, rel=1e-12)


@pytest.mark.finding("10.18")
def test_split_factor_uses_the_solved_source_current() -> None:
    """The denominator is the source current of the solve (snapshot), not
    the source as the world holds it afterwards."""
    w = _chain_world()
    res = _solve(w)
    s_before = gf.split_factor(res, w, frequency_index=1)
    w.sources[0].magnitude = 7.0  # world edited after the solve
    assert gf.split_factor(res, w, frequency_index=1) == pytest.approx(
        s_before, rel=1e-15
    )
