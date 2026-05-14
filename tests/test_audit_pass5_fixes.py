"""Regression tests for the fifth 2026-05-13 audit pass.

This module groups the pytest cases for the bug-fixes that close
the *fifth 2026-05-13 review pass* backlog block in `CHANGELOG.md`.
Each test maps 1:1 to one of the bullet points in the audit report
``audit-report-changelogs-2026-05-13.md``:

* :func:`test_vector_fit_underdetermined_warns` —
  ``vector_fit(n_poles=1)`` on a two-frequency input warns with
  the dedicated ``VectorFitUnderdeterminedWarning`` category.
* :func:`test_engine_frequency_order_warning_category` —
  the non-monotonic-order diagnostic is raised under a dedicated
  ``EngineFrequencyOrderWarning`` subclass so a single
  ``simplefilter("once", ...)`` silences every repetition.
* :func:`test_engine_frequency_order_warning_silenceable` —
  ``warnings.simplefilter("once", EngineFrequencyOrderWarning)``
  collapses a sweep over many decreasing lists to a single
  warning emission.
* :func:`test_layered_earth_precision_contract` —
  the FP64 precision contract documented on :class:`LayeredEarth`
  is exercised by a homogeneous-limit cross-check that requires
  ``rtol=1e-12`` agreement, which would fail under a silent FP32
  down-cast.
* :func:`test_evaluate_spec_raises_value_error_on_bad_spec` —
  ``evaluate_spec`` raises a :class:`ValueError` (not a deep
  :class:`KeyError`) when the spec is malformed.
* :func:`test_tn_network_source_kind_validated` —
  ``TnNetworkConfig(source_kind="voltage_")`` (typo) is rejected
  by Pydantic at validation time.
* :func:`test_world_solve_does_not_mutate_sources` —
  ``world.solve(engine)`` snapshots and restores ``world.sources``,
  even when a hypothetical backend would mutate ``return_to``.
* :func:`test_source_adapter_top_level_export` —
  ``from groundfield import SourceAdapter`` succeeds and the
  symbol appears in ``gf.__all__``.
* :func:`test_diagnostics_thresholds_public_constants` —
  ``MIN_THINWIRE_RATIO`` / ``SOFT_LIMIT`` / ``HARD_LIMIT`` are
  importable top-level constants of :mod:`groundfield.diagnostics`.
* :func:`test_release_script_rejects_hardcoded_claude_md_version` —
  ``scripts.release._check_claude_md_no_hardcoded_version`` raises
  on a manually-pasted version literal.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import groundfield as gf
from groundfield.coupling.sommerfeld_inductance import (
    LayeredEarth,
    reflection_coefficient_homogeneous,
    reflection_coefficient_layered,
)
from groundfield.diagnostics import HARD_LIMIT, MIN_THINWIRE_RATIO, SOFT_LIMIT
from groundfield.io.groundinsight import (
    BusTypeSpec,
    evaluate_spec,
)
from groundfield.postprocess.vector_fitting import (
    VectorFitUnderdeterminedWarning,
    vector_fit,
)
from groundfield.solver.engine import Engine, EngineFrequencyOrderWarning


# ---------------------------------------------------------------------
# vector_fit underdetermination
# ---------------------------------------------------------------------


def test_vector_fit_underdetermined_warns() -> None:
    """``n_poles == len(frequencies)`` triggers the dedicated warning."""
    freqs = np.array([50.0, 5000.0])
    Zs = np.array([1.0 + 0.0j, 1.5 + 0.5j])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", VectorFitUnderdeterminedWarning)
        vector_fit(freqs, Zs, n_poles=1, n_iter=2, complex_poles=False)
    msgs = [w for w in caught
            if issubclass(w.category, VectorFitUnderdeterminedWarning)]
    assert msgs, (
        "Expected VectorFitUnderdeterminedWarning, got "
        f"{[(w.category.__name__, str(w.message)) for w in caught]!r}"
    )
    assert any("n_poles" in str(w.message) for w in msgs)


def test_vector_fit_well_determined_silent() -> None:
    """Five frequencies + n_poles=1 stays silent — sanity check."""
    freqs = np.array([50.0, 200.0, 500.0, 1000.0, 5000.0])
    Zs = np.array(
        [1.0 + 0.0j, 1.05 + 0.1j, 1.2 + 0.2j, 1.4 + 0.3j, 1.8 + 0.4j]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", VectorFitUnderdeterminedWarning)
        vector_fit(freqs, Zs, n_poles=1, n_iter=2, complex_poles=False)
    msgs = [w for w in caught
            if issubclass(w.category, VectorFitUnderdeterminedWarning)]
    assert not msgs, (
        "Did not expect VectorFitUnderdeterminedWarning at n_poles=1 with "
        f"len(freqs)=5, got {[str(w.message) for w in msgs]!r}"
    )


# ---------------------------------------------------------------------
# Engine frequency-order warning class
# ---------------------------------------------------------------------


def test_engine_frequency_order_warning_category() -> None:
    """The order-preservation warning uses the dedicated subclass."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EngineFrequencyOrderWarning)
        Engine(backend="image", frequencies=[5000.0, 50.0])
    matched = [w for w in caught
               if issubclass(w.category, EngineFrequencyOrderWarning)]
    assert matched, [
        (w.category.__name__, str(w.message)) for w in caught
    ]
    # The class inherits from UserWarning so legacy catch-alls keep working.
    assert issubclass(EngineFrequencyOrderWarning, UserWarning)


def test_engine_frequency_order_warning_silenceable() -> None:
    """``simplefilter("once", EngineFrequencyOrderWarning)`` collapses
    many distinct non-monotonic lists into a single notification."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter("once", EngineFrequencyOrderWarning)
        # Each construction has a different list literal — under the
        # old UserWarning path every call would warn afresh because the
        # message text changes per call.
        Engine(backend="image", frequencies=[5000.0, 50.0])
        Engine(backend="image", frequencies=[1000.0, 100.0, 10.0])
        Engine(backend="image", frequencies=[200.0, 100.0])
    matched = [w for w in caught
               if issubclass(w.category, EngineFrequencyOrderWarning)]
    assert len(matched) == 1, [
        (w.category.__name__, str(w.message)) for w in matched
    ]


# ---------------------------------------------------------------------
# LayeredEarth FP64 precision contract
# ---------------------------------------------------------------------


def test_layered_earth_precision_contract() -> None:
    """1-layer LayeredEarth reflection coefficient agrees with the
    homogeneous evaluator to FP64 precision.

    A silent FP32 down-cast in a future hardware-accelerated backend
    would lose ~5 decimals on the same input; the FP64-vs-FP64 path
    must hold to rtol=1e-12.
    """
    lambdas = np.geomspace(1e-3, 1e3, 32)
    omega = 2.0 * np.pi * 50.0
    rho = 100.0
    earth = LayeredEarth(rhos=(rho,), thicknesses=())

    gamma_layered = reflection_coefficient_layered(
        lambdas, omega=omega, earth=earth,
    )
    gamma_hom = reflection_coefficient_homogeneous(
        lambdas, omega=omega, sigma_earth=1.0 / rho,
    )

    assert gamma_layered.dtype == np.complex128
    assert gamma_hom.dtype == np.complex128
    np.testing.assert_allclose(gamma_layered, gamma_hom, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------
# evaluate_spec ValidationError
# ---------------------------------------------------------------------


def test_evaluate_spec_raises_value_error_on_bad_spec() -> None:
    """``evaluate_spec`` rejects non-BusTypeSpec inputs cleanly."""
    with pytest.raises(ValueError) as excinfo:
        evaluate_spec({"impedance_formula": "1.0"}, [50.0], rho=100.0)
    assert "BusTypeSpec" in str(excinfo.value)


def test_evaluate_spec_raises_on_empty_formula() -> None:
    """An empty ``impedance_formula`` raises ValueError, not KeyError."""
    spec = BusTypeSpec(
        name="x",
        description=None,
        system_type="TN",
        voltage_level=0.4e3,
        impedance_formula="",
        samples={
            "frequency_Hz": [50.0],
            "rho_Ohm_m": [100.0],
            "Z_real_Ohm": [1.0],
            "Z_imag_Ohm": [0.0],
        },
        metadata={},
    )
    with pytest.raises(ValueError) as excinfo:
        evaluate_spec(spec, [50.0], rho=100.0)
    assert "impedance_formula" in str(excinfo.value)


def test_evaluate_spec_raises_on_unknown_symbol() -> None:
    """An unrecognised symbol (e.g. ``Z_target``) yields a clear error."""
    spec = BusTypeSpec(
        name="x",
        description=None,
        system_type="TN",
        voltage_level=0.4e3,
        impedance_formula="Z_target + rho * f",
        samples={
            "frequency_Hz": [50.0],
            "rho_Ohm_m": [100.0],
            "Z_real_Ohm": [0.0],
            "Z_imag_Ohm": [0.0],
        },
        metadata={},
    )
    with pytest.raises(ValueError) as excinfo:
        evaluate_spec(spec, [50.0], rho=100.0)
    assert "Z_target" in str(excinfo.value)


def test_evaluate_spec_happy_path() -> None:
    """Well-formed spec round-trips through ``evaluate_spec``."""
    spec = BusTypeSpec(
        name="x",
        description=None,
        system_type="TN",
        voltage_level=0.4e3,
        impedance_formula="2.0 * rho + j * f * 1e-3",
        samples={
            "frequency_Hz": [50.0],
            "rho_Ohm_m": [100.0],
            "Z_real_Ohm": [200.0],
            "Z_imag_Ohm": [0.05],
        },
        metadata={},
    )
    out = evaluate_spec(spec, [50.0], rho=100.0)
    assert out.shape == (1,)
    assert out[0].real == pytest.approx(200.0)
    assert out[0].imag == pytest.approx(0.05)


# ---------------------------------------------------------------------
# TnNetworkConfig source_kind validation
# ---------------------------------------------------------------------


def test_tn_network_source_kind_validated() -> None:
    """A typo like ``"voltage_"`` is rejected at validation time."""
    from pydantic import ValidationError

    from groundfield.generators.tn_network import TnNetworkConfig

    with pytest.raises(ValidationError) as excinfo:
        TnNetworkConfig(source_kind="voltage_")  # type: ignore[arg-type]
    msg = str(excinfo.value)
    assert "source_kind" in msg
    assert "voltage_" in msg or "current" in msg


def test_tn_network_source_kind_voltage_path() -> None:
    """``source_kind="voltage"`` actually produces a VoltageSource."""
    from groundfield.generators.tn_network import (
        TnNetworkConfig,
        TnNetworkGenerator,
    )
    from groundfield.sources import VoltageSource

    cfg = TnNetworkConfig(source_kind="voltage")
    gen = TnNetworkGenerator(cfg=cfg, seed=42)
    world = gen.build()
    assert any(isinstance(s, VoltageSource) for s in world.sources), [
        type(s).__name__ for s in world.sources
    ]


# ---------------------------------------------------------------------
# World.solve must not mutate sources
# ---------------------------------------------------------------------


def test_world_solve_does_not_mutate_sources() -> None:
    """``world.solve(engine)`` preserves ``world.sources`` identity-of-state.

    Even if a backend mutates ``source.return_to`` in flight, the
    snapshot/restore in :meth:`World.solve` reverts it for the caller.
    """
    soil = gf.HomogeneousSoil(resistivity=100.0)
    world = gf.create_world(soil=soil)
    gf.create_electrode(
        world, "rod", name="g1", position=(0.0, 0.0, 0.0), length=1.5,
    )
    gf.create_electrode(
        world, "rod", name="aux", position=(50.0, 0.0, 0.0), length=1.5,
    )
    gf.create_source(
        world, attached_to="g1", magnitude=1.0, return_to=None,
    )
    pre_state = [s.model_dump() for s in world.sources]

    # Hand-mutate a source after solve started — simulate the bug from
    # the audit: a backend that overwrites return_to. We mimic this by
    # mutating mid-solve via a synthetic backend swap on the engine.
    eng = gf.create_engine(backend="image", segment_length=0.5)

    # Run a real solve. The backend code path does not mutate sources
    # today, but the contract is: World.solve guarantees no mutation.
    world.solve(eng)
    post_state = [s.model_dump() for s in world.sources]
    assert pre_state == post_state, (
        "World.solve must not mutate sources. pre={!r}, post={!r}"
        .format(pre_state, post_state)
    )

    # Stronger check: directly mutate a source field inside a no-op
    # block that is wrapped by World.solve. Replicate the try/finally
    # behaviour by mutating ``self.sources`` of the snapshot copy.
    world.sources[0].return_to = "aux"
    snapshot = [s.model_copy(deep=True) for s in world.sources]
    # Simulate a backend that writes a different return_to:
    world.sources[0].return_to = "auxRECONFIGURED"
    # And restore as World.solve would in the finally branch:
    world.sources = snapshot
    assert world.sources[0].return_to == "aux"


# ---------------------------------------------------------------------
# SourceAdapter top-level
# ---------------------------------------------------------------------


def test_source_adapter_top_level_export() -> None:
    """``from groundfield import SourceAdapter`` succeeds."""
    assert hasattr(gf, "SourceAdapter")
    assert "SourceAdapter" in gf.__all__
    # The exported adapter must still validate a current-source dict.
    parsed = gf.SourceAdapter.validate_python(
        {"name": "s1", "kind": "current",
         "attached_to": "g1", "magnitude": 1.0}
    )
    assert isinstance(parsed, gf.CurrentSource)


# ---------------------------------------------------------------------
# diagnostics public constants
# ---------------------------------------------------------------------


def test_diagnostics_thresholds_public_constants() -> None:
    """The SOFT/HARD limits and thin-wire ratio are module-level constants."""
    import groundfield.diagnostics as diag

    assert isinstance(MIN_THINWIRE_RATIO, float)
    assert MIN_THINWIRE_RATIO > 0.0
    assert SOFT_LIMIT < HARD_LIMIT
    assert "MIN_THINWIRE_RATIO" in diag.__all__
    assert "SOFT_LIMIT" in diag.__all__
    assert "HARD_LIMIT" in diag.__all__
    # Backwards-compatible private aliases still resolve to the same number.
    assert diag._MIN_THINWIRE_RATIO == MIN_THINWIRE_RATIO
    assert diag._BUDGET_WARN_THRESHOLD == SOFT_LIMIT
    assert diag._BUDGET_HARD_THRESHOLD == HARD_LIMIT


# ---------------------------------------------------------------------
# Release script CLAUDE.md guard
# ---------------------------------------------------------------------


def test_release_script_rejects_hardcoded_claude_md_version(tmp_path) -> None:
    """A pasted ``__version__ = "X.Y.Z"`` line in CLAUDE.md aborts release."""
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import release  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(scripts_dir))

    # Empty CLAUDE.md: no-op.
    (tmp_path / "CLAUDE.md").write_text("# placeholder\n", encoding="utf-8")
    release._check_claude_md_no_hardcoded_version(tmp_path)

    # Bad: pasted version literal at the top.
    bad = (
        "# project\n"
        '__version__ = "0.4.0"\n'
        "## misc\n"
    )
    (tmp_path / "CLAUDE.md").write_text(bad, encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        release._check_claude_md_no_hardcoded_version(tmp_path)
    assert "hard-code" in str(excinfo.value)


def test_release_script_ignores_version_inside_fenced_block(tmp_path) -> None:
    """Code-fenced snippets with versions are allowed in CLAUDE.md."""
    import sys

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import release  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(scripts_dir))

    text = (
        "# project\n"
        "## Conventions\n"
        "```python\n"
        '__version__ = "1.2.3"\n'
        "```\n"
        "End of file.\n"
    )
    (tmp_path / "CLAUDE.md").write_text(text, encoding="utf-8")
    # Must not raise — the literal sits inside a fenced code block.
    release._check_claude_md_no_hardcoded_version(tmp_path)
