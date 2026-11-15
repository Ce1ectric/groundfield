"""The effective-resistivity preset in the BusType document (user decision Q39).

A BusType fitted over ``effective_resistivity(soil, preset=...)`` records the
preset as ``metadata["rho_preset"]`` (``None`` when none is named); documents
of earlier versions load with ``None``. ``evaluate_spec`` and
``bustype_from_spec`` take the preset of the bus and warn when the document
names another one: a preset holds for its electrode class only (the AP1
station with ``"lv_group_4_p40"`` misses the field solution by 39.9 % rms).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

import groundfield as gf
from groundfield.io.groundinsight import (
    BusTypeSpec,
    RhoPresetMismatchWarning,
    bustype_from_spec,
    evaluate_spec,
    fit_quality_summary,
    load_bustype_json,
    save_bustype_json,
    to_bustype_dict,
)
from tests.test_io_groundinsight import _make_rho_f_fit, _make_vector_fit

pytestmark = pytest.mark.finding("10.84")


def _payload(kind: str, **kwargs) -> dict:
    # user decision Q39
    if kind == "vector_fit":
        fit, extra = _make_vector_fit(), {"rho_at_fit": 83.705}
    else:
        fit, extra = _make_rho_f_fit(), {}
    return to_bustype_dict(
        fit, name=kind, system_type="TN", voltage_level=0.4, **extra, **kwargs
    )


@pytest.mark.parametrize("kind", ["vector_fit", "rho_f_standard"])
def test_the_document_records_the_preset(kind: str, tmp_path) -> None:
    payload = _payload(kind, rho_preset="station")
    md = payload["metadata"]
    assert md["rho_preset"] == "station"
    keys = list(md)
    assert keys.index("rho_preset") == keys.index("rho_definition") + 1
    json.dumps(payload, allow_nan=False)
    path = tmp_path / f"{kind}.json"
    fit = _make_vector_fit() if kind == "vector_fit" else _make_rho_f_fit()
    extra = {"rho_at_fit": 83.705} if kind == "vector_fit" else {}
    save_bustype_json(
        fit,
        path,
        name=kind,
        system_type="TN",
        voltage_level=0.4,
        rho_preset="lv_group_2_p10",
        **extra,
    )
    spec = load_bustype_json(path)
    assert spec.rho_preset == "lv_group_2_p10"
    assert BusTypeSpec.from_dict(spec.to_dict()).rho_preset == "lv_group_2_p10"
    assert "rho_preset=lv_group_2_p10" in fit_quality_summary(spec)


@pytest.mark.parametrize("kind", ["vector_fit", "rho_f_standard"])
def test_no_preset_is_recorded_as_null(kind: str) -> None:
    payload = _payload(kind)
    assert payload["metadata"]["rho_preset"] is None
    assert BusTypeSpec.from_dict(payload).rho_preset is None


def test_documents_without_the_entry_load_with_none() -> None:
    """Documents of 0.15.0 (no rho_definition) and of 2026.11.0 (no
    rho_preset) stay loadable."""
    payload = _payload("rho_f_standard")
    del payload["metadata"]["rho_preset"]
    spec = BusTypeSpec.from_dict(payload)
    assert spec.rho_preset is None
    del payload["metadata"]["rho_definition"]
    assert BusTypeSpec.from_dict(payload).rho_preset is None


@pytest.mark.parametrize("bad", ["lv_netwrk", "Station", "", 3, ["station"]])
def test_an_unknown_preset_is_refused_on_export(bad) -> None:
    with pytest.raises(ValueError, match="rho_preset"):
        _payload("rho_f_standard", rho_preset=bad)


def test_a_misspelt_preset_gets_a_suggestion() -> None:
    with pytest.raises(ValueError, match="lv_network"):
        _payload("rho_f_standard", rho_preset="lv_netwrk")


@pytest.mark.parametrize("bad", [3, ["station"], {"name": "station"}])
def test_a_document_with_a_malformed_preset_is_refused(bad) -> None:
    payload = _payload("rho_f_standard")
    payload["metadata"]["rho_preset"] = bad
    with pytest.raises(ValueError, match="rho_preset"):
        BusTypeSpec.from_dict(payload)


def test_evaluate_warns_when_the_bus_uses_another_preset() -> None:
    """The AP1 station fitted with "station", its bus with "lv_group_4_p40"."""
    spec = BusTypeSpec.from_dict(_payload("rho_f_standard", rho_preset="station"))
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=30.0, h_1=5.0)
    rho_bus = gf.effective_resistivity(soil, preset="lv_group_4_p40")
    with pytest.warns(RhoPresetMismatchWarning, match="lv_group_4_p40"):
        wrong = evaluate_spec(spec, [50.0], rho_bus, rho_preset="lv_group_4_p40")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RhoPresetMismatchWarning)
        rho_ok = gf.effective_resistivity(soil, preset="station")
        right = evaluate_spec(spec, [50.0], rho_ok, rho_preset="station")
        evaluate_spec(spec, [50.0], rho_ok)  # no preset named: nothing to check
    assert not np.allclose(wrong, right)


def test_evaluate_checks_nothing_without_a_recorded_preset() -> None:
    spec = BusTypeSpec.from_dict(_payload("rho_f_standard"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RhoPresetMismatchWarning)
        evaluate_spec(spec, [50.0], 100.0, rho_preset="station")


def test_evaluate_refuses_an_unknown_preset_of_the_caller() -> None:
    spec = BusTypeSpec.from_dict(_payload("rho_f_standard", rho_preset="station"))
    with pytest.raises(ValueError, match="rho_preset"):
        evaluate_spec(spec, [50.0], 100.0, rho_preset="stations")


@pytest.fixture
def groundinsight():
    return pytest.importorskip(
        "groundinsight",
        reason="groundinsight is an optional dependency for the bridge tests",
    )


def test_bustype_from_spec_feeds_a_groundinsight_bus(groundinsight, tmp_path) -> None:
    """Bridge test against the PyPI groundinsight of the dev group: a
    document with a preset, saved and loaded, becomes a live BusType whose
    Bus computes evaluate_spec's impedance to 1e-9; another preset warns."""
    from groundinsight.models.core_models import Bus, BusType

    fit = _make_rho_f_fit()
    path = save_bustype_json(
        fit,
        tmp_path / "station.json",
        name="station",
        system_type="TN",
        voltage_level=20.0,
        rho_preset="station",
    )
    spec = load_bustype_json(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RhoPresetMismatchWarning)
        bustype = bustype_from_spec(spec, rho_preset="station")
    assert isinstance(bustype, BusType)
    assert bustype.impedance_formula == spec.impedance_formula
    freqs = [50.0, 1000.0]
    rho = 83.705
    bus = Bus(
        name="b",
        type=bustype,
        impedance={f: 0.0j for f in freqs},
        specific_earth_resistance=rho,
    )
    bus.calculate_impedance(freqs)
    ref = evaluate_spec(spec, freqs, rho)
    got = np.array([complex(bus.impedance[float(f)]) for f in freqs])
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-9
    with pytest.warns(RhoPresetMismatchWarning):
        bustype_from_spec(spec, rho_preset="lv_network")
    with pytest.raises(TypeError, match="BusTypeSpec"):
        bustype_from_spec(spec.to_dict())
