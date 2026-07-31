"""Regression tests for review pass 9 (2026-07-28), postprocess + io.

Covered findings
----------------
F41
    :func:`groundfield.postprocess.rho_f_standard.rho_f_standard_from_results`
    used to inject ``Z = 0`` for $(\\rho, f)$ samples at which the
    requested electrode carries no current. A vanishing current means
    the driving-point impedance $Z = U/I$ is *undefined*, not zero:
    the injected zero rows pull the least-squares plane of
    $Z = k_1\\rho + (k_2 + jk_3)f + (k_4 + jk_5)f\\rho$ through the
    origin and bias $k_1$ (the Dwight-class DC spreading-resistance
    slope) low by roughly the dead-sample fraction. Dead samples are
    now masked out, with a ``UserWarning``, mirroring
    :func:`groundfield.postprocess.vector_fitting.rho_f_from_field_result`.
F46
    The frozen CSV column tuples named ``electrode`` / ``cluster``
    columns that the writers never emit (they emit ``name`` and
    ``cluster_root``), so every consumer validating a written file
    against the published constants got a ``KeyError``.
F56
    The module docstring swapped the physical meaning of $k_2$
    (soil-independent *resistive* frequency slope, real part) and
    $k_3$ (soil-independent *reactive* / loop-inductance slope,
    imaginary part), and the two identifiability error messages
    named the wrong confounded coefficient pairs.
"""

from __future__ import annotations

import csv as _csv

import numpy as np
import pytest

from groundfield.io import csv as gf_csv
from groundfield.postprocess import rho_f_standard as rfs
from groundfield.postprocess.rho_f_standard import (
    fit_rho_f_standard,
    rho_f_standard_from_results,
)

# True coefficients of the synthetic sweep used below. k1 = 0.5 is the
# number the pre-fix code got wrong (it returned 0.2975).
K_TRUE = dict(k1=0.5, k2=1e-3, k3=2e-3, k4=1e-5, k5=2e-5)
RHOS = [30.0, 100.0, 200.0, 500.0, 1000.0]
FREQS = [0.0, 50.0, 150.0, 250.0, 350.0, 450.0, 550.0]


def _synthetic_Z(rho, f):
    rho = np.asarray(rho, dtype=float)
    f = np.asarray(f, dtype=float)
    return (
        K_TRUE["k1"] * rho
        + (K_TRUE["k2"] + 1j * K_TRUE["k3"]) * f
        + (K_TRUE["k4"] + 1j * K_TRUE["k5"]) * f * rho
    )


def _results_with_dead_dc(dead_frequencies=(0.0,), *, all_dead=False,
                          drop_currents=False):
    """Build FieldResult-like objects for the rho sweep.

    Samples whose frequency is listed in ``dead_frequencies`` carry
    ``I = 0`` (a passive observer / no galvanic source path at DC),
    which is what used to be turned into a ``Z = 0`` sample.
    """
    from groundfield.solver.result import FieldResult

    results = []
    for rho in RHOS:
        f_arr = np.asarray(FREQS, dtype=float)
        Z = _synthetic_Z(rho, f_arr)
        I = np.ones_like(Z, dtype=complex)
        if all_dead:
            I[:] = 0.0
        else:
            for f_dead in dead_frequencies:
                I[f_arr == f_dead] = 0.0
        U = Z * I
        # Keep the (undefined) potential finite where I = 0 so the old
        # code path would really have produced Z = 0 rather than NaN.
        U = np.where(np.abs(I) > 0.0, U, 0.0 + 0.0j)
        currents = {} if drop_currents else {"g1": list(I)}
        results.append(
            FieldResult(
                backend="image",
                frequencies=list(f_arr),
                electrode_potentials={"g1": list(U)},
                electrode_currents=currents,
                soil_resistivity=rho,
            )
        )
    return results


# ---------------------------------------------------------------------
# F41 — dead (rho, f) samples are masked, not injected as Z = 0
# ---------------------------------------------------------------------


def test_f41_dead_dc_column_is_masked_not_zero_injected() -> None:
    """A dead DC column must not bias k1.

    Pre-fix the f = 0 column entered the design matrix as
    Re Z = Im Z = 0 and dragged the fit toward the origin:
    k1 = 0.2975 instead of the true 0.5 (-41 %), with
    ``rms_relative`` = 0.47. Post-fix the dead samples are dropped
    and k1 is recovered exactly.
    """
    results = _results_with_dead_dc()
    with pytest.warns(UserWarning, match="carries no current at 5 of 35"):
        fit = rho_f_standard_from_results(results, RHOS, "g1")

    assert fit.k1 == pytest.approx(K_TRUE["k1"], rel=1e-9), (
        "k1 must be recovered from the live samples; the pre-fix "
        "Z = 0 injection returned 0.2975 for this sweep."
    )
    # Guard the specific pre-fix number so the regression cannot creep
    # back in under a different mechanism.
    assert abs(fit.k1 - 0.2974504249291781) > 0.1
    assert fit.rms_relative < 1e-9, (
        "the fit must be near-exact once the dead rows are gone "
        f"(pre-fix rms_relative was 0.47); got {fit.rms_relative}"
    )


def test_f41_masked_samples_are_absent_from_the_diagnostics() -> None:
    """The dropped samples must not appear in ``sample_*`` either."""
    results = _results_with_dead_dc()
    with pytest.warns(UserWarning):
        fit = rho_f_standard_from_results(results, RHOS, "g1")

    assert fit.sample_f.size == len(RHOS) * (len(FREQS) - 1) == 30
    assert not np.any(fit.sample_f == 0.0)
    assert np.all(np.abs(fit.sample_Z) > 0.0)


def test_f41_all_dead_raises_instead_of_fitting_zeros() -> None:
    """No live sample at all is an error, not an all-zero fit.

    Pre-fix every sample became Z = 0 and the function happily
    returned k1 = ... = k5 = 0 with rms_relative = 0.
    """
    results = _results_with_dead_dc(all_dead=True)
    with pytest.raises(ValueError, match="carries no current at any"):
        rho_f_standard_from_results(results, RHOS, "g1")


def test_f41_electrode_missing_from_currents_raises_helpful_keyerror() -> None:
    """``electrode_name`` is validated against the currents too.

    Pre-fix only ``electrode_potentials`` was checked and the next
    line raised a bare ``KeyError('g1')``.
    """
    results = _results_with_dead_dc(drop_currents=True)
    with pytest.raises(KeyError, match="not in FieldResult currents"):
        rho_f_standard_from_results(results, RHOS, "g1")


def test_f41_all_alive_sweep_is_unchanged_and_silent() -> None:
    """The happy path keeps working and emits no warning."""
    results = _results_with_dead_dc(dead_frequencies=())
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = rho_f_standard_from_results(results, RHOS, "g1")
    assert fit.sample_f.size == len(RHOS) * len(FREQS)
    for key, true_val in K_TRUE.items():
        assert getattr(fit, key) == pytest.approx(true_val, rel=1e-8)


# ---------------------------------------------------------------------
# F56 — k2 is resistive, k3 reactive (docstring / message truth)
# ---------------------------------------------------------------------


def _bullet(doc: str, marker: str) -> str:
    """Return the docstring bullet introduced by ``marker``."""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if marker in line:
            block = [line]
            for nxt in lines[i + 1:]:
                if nxt.strip().startswith("- ") or not nxt.strip():
                    break
                block.append(nxt)
            return " ".join(block).lower()
    raise AssertionError(f"marker {marker!r} not found in module docstring")


def test_f56_module_docstring_assigns_k2_and_k3_correctly() -> None:
    """k2 is the resistive slope, k3 the inductive one.

    Pre-fix the k2 bullet read "purely-inductive coupling" and the k3
    bullet "purely-resistive frequency-dependent term" — exactly
    reversed with respect to ``(k2 + 1j * k3) * f`` in the code.
    """
    doc = rfs.__doc__ or ""
    k2_bullet = _bullet(doc, "$k_2\\,f$")
    k3_bullet = _bullet(doc, "$k_3\\,f$")

    assert "resistive" in k2_bullet and "inductive" not in k2_bullet, k2_bullet
    assert ("reactive" in k3_bullet or "inductance" in k3_bullet), k3_bullet
    assert "purely-resistive" not in k3_bullet, k3_bullet


def test_f56_code_agrees_with_the_documented_split() -> None:
    """The implementation puts k2/k4 in Re Z and k3/k5 in Im Z."""
    fit = rfs.RhoFStandardFit(
        k1=0.0, k2=1.0, k3=2.0, k4=0.0, k5=0.0,
        rms_error=0.0, rms_relative=0.0,
        sample_rho=np.array([1.0]), sample_f=np.array([1.0]),
        sample_Z=np.array([0.0 + 0.0j]),
    )
    Z = complex(fit.evaluate(rho=100.0, f=1.0))
    assert Z.real == pytest.approx(1.0)  # k2 -> resistance
    assert Z.imag == pytest.approx(2.0)  # k3 -> reactance


def test_f56_identifiability_messages_name_the_confounded_pair() -> None:
    """A constant rho confounds k2/k4; a constant f confounds k1/k4.

    Pre-fix the two messages were swapped (rho -> "k1 and k4",
    f -> "k2 and k4").
    """
    freqs = [10.0, 50.0, 150.0, 500.0]
    rhos_const = [100.0] * len(freqs)
    Z = [_synthetic_Z(100.0, f) for f in freqs]
    with pytest.raises(ValueError, match=r"distinct rho values.*k2 and k4"):
        fit_rho_f_standard(rhos_const, freqs, Z)

    rhos = [50.0, 100.0, 500.0, 1000.0]
    f_const = [50.0] * len(rhos)
    Z = [_synthetic_Z(r, 50.0) for r in rhos]
    with pytest.raises(ValueError, match=r"distinct f values.*k1 and k4"):
        fit_rho_f_standard(rhos, f_const, Z)


# ---------------------------------------------------------------------
# F46 — the frozen CSV column tuples match what the writers emit
# ---------------------------------------------------------------------


def _solved_single_rod():
    import groundfield as gf
    from groundfield.geometry.electrodes import RodElectrode
    from groundfield.soil.models import HomogeneousSoil
    from groundfield.sources import CurrentSource

    world = gf.World(
        soil=HomogeneousSoil(resistivity=100.0),
        electrodes=[
            RodElectrode(name="r1", position=(0.0, 0.0, 0.0), length=3.0)
        ],
        sources=[
            CurrentSource(name="s1", attached_to="r1", magnitude=100.0)
        ],
    )
    result = gf.Engine(frequencies=[50.0], backend="image").solve(world)
    return world, result


def test_f46_frozen_constants_have_the_writer_column_names() -> None:
    """The constants must not promise ``electrode`` / ``cluster``."""
    assert gf_csv.ELECTRODE_TABLE_REQUIRED_COLUMNS == (
        "name", "cluster_root", "I_re", "I_im", "abs_I",
    )
    assert gf_csv.CLUSTER_IMPEDANCE_REQUIRED_COLUMNS == (
        "cluster_root", "Z_re", "Z_im", "abs_Z",
    )
    assert "electrode" not in gf_csv.ELECTRODE_TABLE_REQUIRED_COLUMNS
    assert "cluster" not in gf_csv.CLUSTER_IMPEDANCE_REQUIRED_COLUMNS


def test_f46_written_csvs_satisfy_the_published_contract(tmp_path) -> None:
    """Round trip: solve, write, and validate the real header rows.

    Pre-fix ``set(REQUIRED) - set(header)`` was ``{'electrode'}`` for
    the electrode table and ``{'cluster'}`` for the cluster table.
    """
    world, result = _solved_single_rod()

    p_electrodes = gf_csv.save_electrode_table_csv(
        result, tmp_path / "electrodes.csv", world=world
    )
    p_clusters = gf_csv.save_cluster_impedances_csv(
        result, tmp_path / "clusters.csv"
    )

    for path, required in (
        (p_electrodes, gf_csv.ELECTRODE_TABLE_REQUIRED_COLUMNS),
        (p_clusters, gf_csv.CLUSTER_IMPEDANCE_REQUIRED_COLUMNS),
    ):
        with open(path, newline="", encoding="utf-8") as fh:
            header = next(_csv.reader(fh))
        missing = set(required) - set(header)
        assert not missing, f"{path.name} is missing {sorted(missing)}"

    # And the potential-path writer still emits its frozen columns in
    # exactly the frozen order.
    p_path = gf_csv.save_potential_path_csv(
        result, tmp_path / "path.csv", start=(1.0, 0.0, 0.0), distance=5.0, n=3
    )
    with open(p_path, newline="", encoding="utf-8") as fh:
        header = next(_csv.reader(fh))
    assert tuple(header) == gf_csv.POTENTIAL_PATH_COLUMNS


def test_f46_constants_are_usable_as_pandas_keys(tmp_path) -> None:
    """The documented ``groupby``/indexing use case works."""
    pd = pytest.importorskip("pandas")
    world, result = _solved_single_rod()

    p_electrodes = gf_csv.save_electrode_table_csv(
        result, tmp_path / "electrodes.csv", world=world
    )
    df = pd.read_csv(p_electrodes)
    # First entry of the tuple is the identifier column by convention.
    id_col = gf_csv.ELECTRODE_TABLE_REQUIRED_COLUMNS[0]
    assert df.groupby(id_col)["abs_I"].sum().sum() > 0.0

    p_clusters = gf_csv.save_cluster_impedances_csv(
        result, tmp_path / "clusters.csv"
    )
    dfc = pd.read_csv(p_clusters)
    cluster_col = gf_csv.CLUSTER_IMPEDANCE_REQUIRED_COLUMNS[0]
    assert dfc[cluster_col].tolist() == ["r1"]
