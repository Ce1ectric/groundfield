"""Regression tests for the fourth 2026-05-12 audit pass.

This module groups the new pytest cases for the bug-fixes that closed
the *fourth 2026-05-12 review pass* backlog block in `CHANGELOG.md`.
Each test maps 1:1 to one of the bullet points in the audit report
``audit-report-changelogs-2026-05-12-pass4.md``:

* :func:`test_source_union_uses_discriminator` —
  ``Source = Union[CurrentSource, VoltageSource]`` now ships an
  explicit ``Discriminator("kind")`` so error messages point at the
  selected sub-class instead of the union's validator chain.
* :func:`test_set_boundary_conditions_warns_on_revert` —
  reverting a previously-set non-default ``BoundaryConditions`` field
  back to the default emits a ``UserWarning``.
* :func:`test_engine_frequencies_preserve_order_and_warn` —
  :class:`Engine` no longer silently sorts ``frequencies``; a
  non-monotonic input is accepted but raises ``UserWarning``.
* :func:`test_engine_with_frequencies_preserve_order` —
  the opt-in :meth:`Engine.with_frequencies` constructor silences
  the warning.
* :func:`test_tn_network_source_return_to_override_warns` —
  ``TnNetworkConfig.source_return_to`` takes precedence over the
  measurement-setup aux electrode and emits ``UserWarning`` to
  surface the previously-silent override.
* :func:`test_vector_fit_rejects_zero_poles` —
  ``vector_fit(n_poles=0)`` raises ``ValueError``.
* :func:`test_io_csv_column_schema_locked` —
  the CSV-writer column convention is locked via the new
  ``POTENTIAL_PATH_COLUMNS`` / ``ELECTRODE_TABLE_REQUIRED_COLUMNS`` /
  ``CLUSTER_IMPEDANCE_REQUIRED_COLUMNS`` constants.
* :func:`test_top_level_reexports_evaluate_spec_and_friends` —
  ``evaluate_spec``, ``fit_quality_summary`` and
  ``coupling.LayeredEarth`` are reachable as top-level attributes
  and appear in ``__all__``.
* :func:`test_mkdocs_polyfill_removed` —
  the ``polyfill.io`` URL is gone from ``mkdocs.yml``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import groundfield as gf
from groundfield.io import csv as gf_csv
from groundfield.solver.engine import Engine
from groundfield.sources import (
    CurrentSource,
    Source,
    SourceAdapter,
    VoltageSource,
)


# ---------------------------------------------------------------------
# Source discriminated union
# ---------------------------------------------------------------------


def test_source_union_uses_discriminator() -> None:
    """``Source`` round-trips through Pydantic with the right sub-class."""
    # JSON-round-trip via the TypeAdapter
    cs = CurrentSource(name="s1", attached_to="g1", magnitude=1.0)
    payload = cs.model_dump()
    parsed = SourceAdapter.validate_python(payload)
    assert isinstance(parsed, CurrentSource)
    assert parsed.kind == "current"
    assert parsed.magnitude == pytest.approx(1.0)

    vs = VoltageSource(name="v1", attached_to="g1", magnitude=230.0)
    parsed_v = SourceAdapter.validate_python(vs.model_dump())
    assert isinstance(parsed_v, VoltageSource)
    assert parsed_v.kind == "voltage"


def test_source_union_invalid_kind_raises() -> None:
    """An unknown ``kind`` must be rejected by the discriminator."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        SourceAdapter.validate_python(
            {"name": "x", "kind": "no-such-kind",
             "attached_to": "g1", "magnitude": 1.0}
        )
    # The discriminator surfaces the bad ``kind`` value explicitly.
    assert "kind" in str(excinfo.value)


# ---------------------------------------------------------------------
# BoundaryConditions revert warning
# ---------------------------------------------------------------------


def test_set_boundary_conditions_warns_on_revert() -> None:
    """Reverting a non-default value back to the default warns."""
    world = gf.create_world(soil=gf.HomogeneousSoil(resistivity=100.0))

    # Step 1: set a non-default value -> regular non-default warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        world.set_boundary_conditions(surface="dirichlet")
        # At least one warning fires
        assert any("non-default" in str(w.message) for w in caught)

    # Step 2: revert back to the default -> revert warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        world.set_boundary_conditions(surface="neumann")
        revert_msgs = [str(w.message) for w in caught
                       if "reverted to the default" in str(w.message)]
        assert revert_msgs, (
            "Expected a UserWarning on revert, got: "
            f"{[str(w.message) for w in caught]!r}"
        )


# ---------------------------------------------------------------------
# Engine.frequencies order preservation
# ---------------------------------------------------------------------


def test_engine_frequencies_preserve_order_and_warn() -> None:
    """Non-monotonic frequencies are kept in order and warn the user."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        eng = Engine(backend="image", frequencies=[5000.0, 50.0])
    assert eng.frequencies == [5000.0, 50.0]
    assert any(
        "not strictly increasing" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]


def test_engine_with_frequencies_preserve_order() -> None:
    """``with_frequencies(preserve_order=True)`` keeps order and is silent."""
    eng = Engine(backend="image")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        new = eng.with_frequencies(5000.0, 50.0, preserve_order=True)
    assert new.frequencies == [5000.0, 50.0]
    # The receiver must not be mutated.
    assert eng.frequencies == [50.0]


# ---------------------------------------------------------------------
# TnNetworkConfig.source_return_to precedence
# ---------------------------------------------------------------------


def test_tn_network_source_return_to_override_warns() -> None:
    """User-set ``source_return_to`` overrides aux and warns."""
    from groundfield.generators.tn_network import (
        TnNetworkConfig,
        TnNetworkGenerator,
    )
    from groundfield.generators.measurement import (
        MeasurementInjectionConfig,
        MeasurementProbeConfig,
        MeasurementSetupConfig,
    )

    cfg = TnNetworkConfig(
        measurement=MeasurementSetupConfig(
            injection=MeasurementInjectionConfig(),
            probe=MeasurementProbeConfig(),
        ),
        source_return_to="user-specified-electrode",
    )
    gen = TnNetworkGenerator(cfg=cfg, seed=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        world = gen.build()
    msgs = [str(w.message) for w in caught
            if "source_return_to" in str(w.message)]
    assert msgs, [str(w.message) for w in caught]

    # The created source carries the user-specified return_to.
    assert any(
        s.return_to == "user-specified-electrode" for s in world.sources
    ), [s.return_to for s in world.sources]


# ---------------------------------------------------------------------
# vector_fit zero-pole rejection
# ---------------------------------------------------------------------


def test_vector_fit_rejects_zero_poles() -> None:
    """``vector_fit(n_poles=0)`` must raise ValueError with a clear msg."""
    freqs = np.array([50.0, 500.0, 5000.0])
    Zs = np.array([1.0 + 0j, 1.0 + 0.1j, 1.0 + 0.5j])
    with pytest.raises(ValueError) as excinfo:
        gf.vector_fit(freqs, Zs, n_poles=0)
    assert "n_poles" in str(excinfo.value)
    assert "0" in str(excinfo.value) or "at least 1" in str(excinfo.value)


# ---------------------------------------------------------------------
# io.csv column schema lock
# ---------------------------------------------------------------------


def test_io_csv_column_schema_locked() -> None:
    """The frozen column tuples document the writer convention."""
    assert gf_csv.POTENTIAL_PATH_COLUMNS == (
        "s", "x", "y", "z", "frequency_Hz", "phi_re", "phi_im", "abs_phi",
    )
    # Magnitude columns differ per quantity by design (phi / I / Z).
    assert "abs_phi" in gf_csv.POTENTIAL_PATH_COLUMNS
    assert "abs_I" in gf_csv.ELECTRODE_TABLE_REQUIRED_COLUMNS
    assert "abs_Z" in gf_csv.CLUSTER_IMPEDANCE_REQUIRED_COLUMNS


# ---------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------


def test_top_level_reexports_evaluate_spec_and_friends() -> None:
    """``evaluate_spec`` / ``fit_quality_summary`` / ``LayeredEarth``."""
    assert hasattr(gf, "evaluate_spec")
    assert hasattr(gf, "fit_quality_summary")
    assert hasattr(gf, "LayeredEarth")
    for name in ("evaluate_spec", "fit_quality_summary", "LayeredEarth"):
        assert name in gf.__all__, name


# ---------------------------------------------------------------------
# mkdocs polyfill cleanup
# ---------------------------------------------------------------------


def test_mkdocs_polyfill_removed() -> None:
    """The flagged ``polyfill.io`` URL is gone from ``mkdocs.yml``."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "mkdocs.yml").read_text(encoding="utf-8")
    assert "polyfill.io" not in text, (
        "polyfill.io is still referenced in mkdocs.yml — four audit "
        "passes in a row flagged this URL; please drop it."
    )
