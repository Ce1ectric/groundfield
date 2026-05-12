"""Tests for the ``groundfield`` → ``groundinsight`` bridge.

Covers ADR-0008's validation programme:

1. Symbol round-trip — the SymPy formula produced by the exporter
   parses through ``groundinsight``'s validator and evaluates to the
   same numbers as ``fit.evaluate(...)``.
2. JSON round-trip — ``save_bustype_json`` followed by
   ``load_bustype_json`` reproduces every numerical field bit-exact.
3. Optional-dependency hygiene — the JSON path works without the
   ``groundinsight`` import; the live-``BusType`` path raises a clear
   ``ImportError`` when ``groundinsight`` is absent.
4. End-to-end — a small AP1-shaped soil-resistivity sweep produces a
   ``RhoFStandardFit``, exports a ``BusType`` and verifies that
   ``groundinsight``'s evaluator reproduces the fit on the sample
   grid.
"""

from __future__ import annotations

import importlib
import json
import math
import sys

import numpy as np
import pytest

from groundfield.io.groundinsight import (
    BusTypeSpec,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    evaluate_spec,
    fit_quality_summary,
    load_bustype_json,
    save_bustype_json,
    to_bustype,
    to_bustype_dict,
)
from groundfield.postprocess.rho_f_standard import (
    RhoFStandardFit,
    fit_rho_f_standard,
)
from groundfield.postprocess.vector_fitting import vector_fit


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_rho_f_fit() -> RhoFStandardFit:
    """A small but well-conditioned rho-f sample set with known coefficients."""
    rng = np.random.default_rng(0)
    rho_grid = np.array([50.0, 100.0, 300.0, 1000.0])
    f_grid = np.array([1.0, 50.0, 100.0, 500.0, 1000.0])
    rho, f = np.meshgrid(rho_grid, f_grid, indexing="ij")
    rho = rho.ravel()
    f = f.ravel()
    k1, k2, k3, k4, k5 = 0.040, 1.5e-4, 0.0, 1.0e-6, 2.5e-7
    Z = (
        k1 * rho
        + (k2 + 1j * k3) * f
        + (k4 + 1j * k5) * f * rho
    )
    Z = Z + (rng.normal(scale=1e-6, size=Z.shape)
             + 1j * rng.normal(scale=1e-6, size=Z.shape))
    return fit_rho_f_standard(rho, f, Z)


def _make_vector_fit():
    """A simple two-real-pole vector fit at fixed soil."""
    R_inf = 25.0
    poles = [-2.0 * math.pi * 50.0, -2.0 * math.pi * 500.0]
    residues = [-2.0 * math.pi * 50.0 * 8.0,
                -2.0 * math.pi * 500.0 * 3.0]
    freqs = np.geomspace(1.0, 1000.0, 25)
    s = 2j * math.pi * freqs
    Z = np.full_like(s, complex(R_inf), dtype=complex)
    for p, r in zip(poles, residues):
        Z = Z + r / (s - p)
    return vector_fit(
        freqs, Z, n_poles=2, n_iter=10,
        include_R_inf=True, include_L_inf=False, complex_poles=False,
    )


# ---------------------------------------------------------------------
# 1. Schema and dict shape
# ---------------------------------------------------------------------


def test_to_bustype_dict_has_schema_v1() -> None:
    fit = _make_rho_f_fit()
    payload = to_bustype_dict(
        fit,
        name="t1", system_type="LV", voltage_level=0.4,
        electrode_name="trafo_ring",
        soil_summary="2-layer rho_1 sweep, rho_2=20, h_1=2.0",
    )
    assert payload["schema"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["name"] == "t1"
    assert payload["system_type"] == "LV"
    assert payload["voltage_level"] == 0.4
    assert "rho" in payload["impedance_formula"]
    assert "f" in payload["impedance_formula"]
    # Samples block: four parallel arrays of equal length.
    samples = payload["samples"]
    n = len(samples["frequency_Hz"])
    assert n > 0
    assert len(samples["rho_Ohm_m"]) == n
    assert len(samples["Z_real_Ohm"]) == n
    assert len(samples["Z_imag_Ohm"]) == n
    # Metadata
    md = payload["metadata"]
    assert md["fit_method"] == "rho_f_standard"
    assert "k1" in md["coefficients"]
    assert md["source"].startswith("groundfield")


def test_vector_fit_export_requires_rho_at_fit() -> None:
    vf = _make_vector_fit()
    with pytest.raises(ValueError, match="rho_at_fit"):
        to_bustype_dict(
            vf, name="vf", system_type="LV", voltage_level=0.4,
        )


def test_vector_fit_export_renders_in_f() -> None:
    import sympy as sp

    vf = _make_vector_fit()
    payload = to_bustype_dict(
        vf, name="vf", system_type="LV", voltage_level=0.4,
        rho_at_fit=100.0, electrode_name="rod_1",
        soil_summary="HomogeneousSoil rho=100",
    )
    formula = payload["impedance_formula"]
    # Vector fit is in s; the export must substitute s -> j * 2*pi * f
    # symbolically. Verify that 'f' appears as a free symbol and that
    # 's' is gone.
    free = {str(sym) for sym in sp.sympify(formula).free_symbols}
    assert "f" in free, f"missing free symbol 'f' in {formula!r}"
    assert "s" not in free, f"residual free 's' in {formula!r}"
    md = payload["metadata"]
    assert md["fit_method"] == "vector_fit"
    assert md["n_poles"] == 2
    assert md["rho_at_fit_Ohm_m"] == 100.0
    assert len(md["poles"]) == 2


# ---------------------------------------------------------------------
# 2. JSON round-trip
# ---------------------------------------------------------------------


def test_json_roundtrip_preserves_payload(tmp_path) -> None:
    fit = _make_rho_f_fit()
    path = tmp_path / "bustype.json"
    save_bustype_json(
        fit, path,
        name="house", system_type="LV", voltage_level=0.4,
        description="single-family house, AP1 reference",
        electrode_name="rod_1",
    )
    spec = load_bustype_json(path)
    assert isinstance(spec, BusTypeSpec)
    assert spec.name == "house"
    assert spec.system_type == "LV"
    assert spec.voltage_level == 0.4

    # Bit-exact numerical round-trip on the samples
    payload = json.loads(path.read_text(encoding="utf-8"))
    np.testing.assert_array_equal(
        spec.samples["frequency_Hz"], payload["samples"]["frequency_Hz"],
    )
    np.testing.assert_array_equal(
        spec.samples["Z_real_Ohm"], payload["samples"]["Z_real_Ohm"],
    )
    # Formula round-trip is verbatim
    assert spec.impedance_formula == payload["impedance_formula"]


def test_load_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "schema": "something.else", "schema_version": 1,
        "name": "x", "system_type": "LV", "voltage_level": 0.4,
        "impedance_formula": "1.0",
        "samples": {}, "metadata": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema name"):
        load_bustype_json(path)


def test_load_rejects_future_schema_version(tmp_path) -> None:
    path = tmp_path / "future.json"
    path.write_text(json.dumps({
        "schema": SCHEMA_NAME, "schema_version": SCHEMA_VERSION + 99,
        "name": "x", "system_type": "LV", "voltage_level": 0.4,
        "impedance_formula": "1.0",
        "samples": {}, "metadata": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_bustype_json(path)


# ---------------------------------------------------------------------
# 3. Symbol round-trip via groundinsight's evaluator
# ---------------------------------------------------------------------


groundinsight = pytest.importorskip(
    "groundinsight",
    reason="groundinsight is an optional dependency for the bridge tests",
)


def test_rho_f_standard_formula_evaluates_through_groundinsight() -> None:
    """The exported formula must reproduce the fit at the sample points
    when fed through ``groundinsight.compute_impedance``."""
    from groundinsight.utils.impedance_calculator import compute_impedance

    fit = _make_rho_f_fit()
    payload = to_bustype_dict(
        fit, name="t", system_type="LV", voltage_level=0.4,
    )
    formula = payload["impedance_formula"]

    # Pick an arbitrary sub-grid covered by the fit's sample set.
    f_grid = sorted(set(fit.sample_f.tolist()))
    rho_test = float(np.unique(fit.sample_rho)[0])
    Z_gi = compute_impedance(formula, f_grid, {"rho": rho_test})
    Z_gi_arr = np.array([
        complex(Z_gi[f].real, Z_gi[f].imag) for f in f_grid
    ])
    # fit.evaluate reproduces Z(rho_test, f) from the same coefficients
    Z_ref = fit.evaluate(rho_test, np.asarray(f_grid))
    rel = np.max(np.abs(Z_gi_arr - Z_ref)) / np.max(np.abs(Z_ref))
    assert rel < 1e-9, f"groundinsight evaluator drifted: rel={rel}"


def test_to_bustype_returns_groundinsight_bustype() -> None:
    """The Python-API path returns a fully validated BusType."""
    from groundinsight.models.core_models import BusType

    fit = _make_rho_f_fit()
    bustype = to_bustype(
        fit, name="t2", system_type="LV", voltage_level=0.4,
    )
    assert isinstance(bustype, BusType)
    assert bustype.name == "t2"
    assert bustype.system_type == "LV"
    assert bustype.voltage_level == 0.4
    # The validator on the formula has already run inside Pydantic.
    assert "rho" in bustype.impedance_formula


def test_vector_fit_formula_evaluates_through_groundinsight() -> None:
    """Vector-fit export with s -> j*2*pi*f must reproduce the fit."""
    from groundinsight.utils.impedance_calculator import compute_impedance

    vf = _make_vector_fit()
    payload = to_bustype_dict(
        vf, name="vf2", system_type="LV", voltage_level=0.4,
        rho_at_fit=100.0,
    )
    formula = payload["impedance_formula"]

    freqs = vf.fit_frequencies.tolist()
    Z_gi = compute_impedance(formula, freqs, {"rho": 100.0})
    Z_gi_arr = np.array([
        complex(Z_gi[float(f)].real, Z_gi[float(f)].imag) for f in freqs
    ])
    Z_ref = vf.evaluate(freqs)
    rel = np.max(np.abs(Z_gi_arr - Z_ref)) / np.max(np.abs(Z_ref))
    # Vector fit was tight (rms ~ 1e-9); allow a small rendering margin.
    assert rel < 1e-3, f"vector-fit formula drift: rel={rel}"


def test_end_to_end_bustype_into_groundinsight_bus() -> None:
    """Build a Bus from the exported BusType and check it computes
    impedance over a frequency list without errors."""
    from groundinsight.models.core_models import Bus

    fit = _make_rho_f_fit()
    bustype = to_bustype(
        fit, name="ap1_house", system_type="LV", voltage_level=0.4,
    )
    rho_for_bus = 100.0
    freqs = [50.0, 100.0, 1000.0]
    bus = Bus(
        name="bus_1", type=bustype,
        impedance={f: 0.0 + 0.0j for f in freqs},
        specific_earth_resistance=rho_for_bus,
    )
    bus.calculate_impedance(freqs)
    assert set(bus.impedance.keys()) == set(float(f) for f in freqs)
    # The Bus impedance must match fit.evaluate at rho_for_bus
    for f in freqs:
        z_ref = complex(fit.evaluate(rho_for_bus, f))
        z_gi = complex(bus.impedance[float(f)].real, bus.impedance[float(f)].imag)
        rel = abs(z_gi - z_ref) / abs(z_ref)
        assert rel < 1e-9, f"Bus impedance drifted at {f} Hz: rel={rel}"


# ---------------------------------------------------------------------
# 4. evaluate_spec sanity (no groundinsight needed)
# ---------------------------------------------------------------------


def test_evaluate_spec_matches_fit_evaluate() -> None:
    fit = _make_rho_f_fit()
    payload = to_bustype_dict(
        fit, name="t3", system_type="LV", voltage_level=0.4,
    )
    spec = BusTypeSpec.from_dict(payload)
    f_test = np.array([10.0, 100.0, 1000.0])
    rho_test = 100.0
    Z_spec = evaluate_spec(spec, f_test, rho_test)
    Z_ref = fit.evaluate(rho_test, f_test)
    np.testing.assert_allclose(Z_spec, Z_ref, rtol=1e-9, atol=1e-12)


def test_fit_quality_summary_is_human_readable() -> None:
    fit = _make_rho_f_fit()
    payload = to_bustype_dict(
        fit, name="t4", system_type="LV", voltage_level=0.4,
    )
    spec = BusTypeSpec.from_dict(payload)
    summary = fit_quality_summary(spec)
    assert "BusTypeSpec" in summary
    assert "rho_f_standard" in summary
    assert "rms" in summary


# ---------------------------------------------------------------------
# 5. Optional-dependency hygiene
# ---------------------------------------------------------------------


def test_to_bustype_raises_clear_error_when_groundinsight_missing(
    monkeypatch,
) -> None:
    """Simulate ``groundinsight`` being unimportable and assert the
    Python-API path raises a clear ImportError; the JSON path keeps
    working."""
    # Make import groundinsight fail.
    real_import = __builtins__["__import__"] if isinstance(
        __builtins__, dict
    ) else __import__

    def fake_import(name, *args, **kwargs):
        if name == "groundinsight" or name.startswith("groundinsight."):
            raise ImportError("simulated missing groundinsight")
        return real_import(name, *args, **kwargs)

    # Drop any cached groundinsight module so the next import goes
    # through fake_import.
    for mod in list(sys.modules):
        if mod == "groundinsight" or mod.startswith("groundinsight."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    monkeypatch.setattr("builtins.__import__", fake_import)

    fit = _make_rho_f_fit()
    # JSON path stays functional
    payload = to_bustype_dict(
        fit, name="x", system_type="LV", voltage_level=0.4,
    )
    assert payload["schema"] == SCHEMA_NAME
    # Python-API path must raise ImportError with a clear hint
    with pytest.raises(ImportError, match="groundinsight"):
        to_bustype(fit, name="x", system_type="LV", voltage_level=0.4)
