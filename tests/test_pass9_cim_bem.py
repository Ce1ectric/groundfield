"""Regression tests for review pass 9, findings F17 / F34 / F39.

Physical / mathematical background
----------------------------------
The ``cim`` backend approximates the upward-looking reflection
coefficient of a stratified half-space by a finite sum of decaying
exponentials,

    Γ_1(λ) ≈ Σ_k a_k · exp(-2 λ β_k),   Re{β_k} > 0,

which turns the layered Sommerfeld integral into a closed-form sum of
complex images. A sum of *decaying* exponentials vanishes for
λ → ∞, whereas the true Γ_1 tends to the top-interface Fresnel
coefficient

    K_1 = (ρ_2 - ρ_1) / (ρ_2 + ρ_1) ≠ 0,

so the asymptote has to be split off analytically and carried by a
β = 0 image (Li et al. 2006, Dan et al. 2021). Up to 0.14.1 the
implementation fitted Γ_1 itself and its pole filters discarded
precisely that constant term, leaving a residual of order |K_1| for
every stack — silently (F39). Meanwhile no solve path consumed the
fit at all: ``cim``/``bem`` reject n ≥ 3 and use exact closed-form
self-kernels for n ≤ 2, yet the failed fit was computed on every
solve and advertised as ``cim_n_images`` / ``cim_rms`` (F34), and the
documentation still routed multi-layer users to ``cim``/``bem``
(F17).

What these tests pin down
-------------------------
F39 — the fit represents the λ → ∞ asymptote, its residual is
      always finite, and failure is reported (``converged=False`` plus
      ``ComplexImageFitWarning``) instead of hidden.
F34 — no fit is evaluated on any solve path, the metadata says so
      (``cim_fit_used=False``, ``cim_rms=None``, ``reduces_to``), and
      the dead n ≥ 3 code paths are gone / raise.
F17 — the engine documentation describes the code: n ≥ 3 routes to
      ``mom_sommerfeld``, and ``cim``/``bem`` are flagged as
      *not* independent cross-checks for n ≤ 2.

Pre-fix behaviour (v0.14.1, ``5ba740d``) that these tests catch:
``fit_complex_images`` on ``LayerStack([100, 400, 50], [2, 3])``
reported ``rms = 0.5847`` against ``|K_1| = 0.600`` and returned
``P = 3``; on a two-layer stack it returned ``P = 0`` with
``rms = nan``; ``cim``/``bem`` result metadata carried
``{'cim_n_images': 0, 'cim_rms': nan}`` and no ``cim_fit_used`` /
``reduces_to`` keys; ``_cim_field_potential`` existed without a
caller.
"""

from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

import groundfield as gf
from groundfield.solver import bem as bem_mod
from groundfield.solver import cim as cim_mod
from groundfield.solver._layered import LayerStack, reflection_gamma
from groundfield.solver.cim import (
    ComplexImageFitWarning,
    fit_complex_images,
)

SEG = 0.1
DOCS = Path(__file__).resolve().parents[1] / "docs"


def _rod_world(soil, *, length: float = 3.0) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "rod", name="g1", position=(0.0, 0.0, 0.0), length=length
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _ring_world(soil) -> gf.World:
    w = gf.create_world(soil=soil)
    gf.create_electrode(
        w, "ring", name="g1", center=(0.0, 0.0, 0.8), radius=2.0
    )
    gf.create_source(w, attached_to="g1", magnitude=1.0)
    return w


def _three_layer_stack(rho_2: float = 400.0) -> LayerStack:
    return LayerStack(
        rhos=np.array([100.0, rho_2, 50.0]), h=np.array([2.0, 3.0])
    )


# =====================================================================
# F39 — the fit must represent the λ → ∞ asymptote of Γ_1
# =====================================================================


@pytest.mark.parametrize("rho_2", [110.0, 200.0, 400.0, 1000.0])
def test_fit_residual_is_far_below_the_asymptote(rho_2: float) -> None:
    """The residual must not track |K_1| (F39).

    Pre-fix numbers (three-layer stack ρ = [100, ρ_2, 50], h = [2, 3]):
    ``rms`` tracked ``|K_1|`` almost exactly — ρ_2 = 110 → 0.0468 vs
    |K_1| = 0.0476; ρ_2 = 400 → 0.5847 vs 0.600. Since |Γ_1| ≤ 1 by
    construction, such a residual means the fit carries no
    information.
    """
    stack = _three_layer_stack(rho_2)
    k_1 = abs(float(stack.K[0]))
    fit = fit_complex_images(stack)

    assert np.isfinite(fit.rms)
    assert fit.converged
    # Two orders of magnitude below the old |K_1| plateau is already a
    # different regime; the measured value is ~1e-13.
    assert fit.rms < 1e-2 * k_1, (
        f"rho_2={rho_2}: rms={fit.rms:.3e} still of order "
        f"|K_1|={k_1:.3e}"
    )
    assert fit.rms < 1e-6


@pytest.mark.parametrize("rho_2", [200.0, 400.0])
def test_fit_two_layer_is_exact_single_asymptote_image(
    rho_2: float,
) -> None:
    """For n = 2, Γ_1 ≡ K_1 and one β = 0 image is exact (F39).

    Pre-fix every two-layer stack returned ``P = 0`` and
    ``rms = nan``: the only pole sits at exactly 1.0 and the
    ``|p| < 0.999`` filter dropped it, i.e. a silent total fit
    failure.
    """
    stack = LayerStack(rhos=np.array([100.0, rho_2]), h=np.array([2.0]))
    fit = fit_complex_images(stack)

    assert fit.a.size == 1
    assert fit.beta[0] == 0.0
    assert fit.a[0].real == pytest.approx(float(stack.K[0]), rel=1e-12)
    assert fit.k_inf == pytest.approx(float(stack.K[0]), rel=1e-12)
    assert fit.rms == 0.0
    assert fit.converged


@pytest.mark.parametrize(
    "h", [[2.0, 3.0], [0.5, 10.0], [10.0, 0.5], [1.0, 1.0]]
)
def test_fit_reproduces_gamma_including_the_asymptote(
    h: list[float],
) -> None:
    """The model must approach K_1 for λ → ∞ (F39).

    Pre-fix the model was a pure sum of decaying exponentials, so it
    approached 0 while Γ_1 approached K_1 = 0.6 — and outside the
    sample window the extrapolation diverged (max |fit - Γ_1| = 126.6
    over λ ∈ [1e-3, 25] for the ρ_2 = 400 stack).

    The ``h = [0.5, 10]`` case additionally covers the grid-scale bug:
    both λ bounds used to be derived from ``h_min``, giving a step of
    1.587 against a structure scale of 0.05, so the pencil stepped
    over the transition entirely.
    """
    stack = LayerStack(
        rhos=np.array([100.0, 400.0, 50.0]), h=np.array(h)
    )
    fit = fit_complex_images(stack)
    k_1 = float(stack.K[0])

    # λ → ∞ limit of the model: only the β = 0 image survives.
    lam_huge = np.array([1e6 / min(h)])
    model_inf = complex(
        (np.exp(-2.0 * lam_huge[:, None] * fit.beta[None, :]) @ fit.a)[0]
    )
    assert model_inf.real == pytest.approx(k_1, abs=1e-9)

    # And the model tracks Γ_1 over a wide, independent λ range.
    lam = np.logspace(-4, 3, 600)
    gamma = reflection_gamma(stack, lam)
    model = np.exp(-2.0 * lam[:, None] * fit.beta[None, :]) @ fit.a
    assert np.max(np.abs(model - gamma)) < 1e-3


def test_fit_never_reports_nan_residual() -> None:
    """A failed fit reports the residual it achieved, not ``nan`` (F39).

    Pre-fix ``rms = float("nan")`` was returned from two early-exit
    branches (all poles filtered, all β non-decaying), and NaN
    propagated into ``result.metadata['cim_rms']``.
    """
    stacks = [
        LayerStack(rhos=np.array([100.0]), h=np.zeros(0)),
        LayerStack(rhos=np.array([100.0, 400.0]), h=np.array([2.0])),
        LayerStack(rhos=np.array([100.0, 100.0]), h=np.array([2.0])),
        _three_layer_stack(),
        LayerStack(
            rhos=np.array([100.0, 100.0, 100.0]), h=np.array([2.0, 3.0])
        ),
        LayerStack(
            rhos=np.array([100.0, 4000.0, 20.0, 900.0]),
            h=np.array([0.2, 0.05, 8.0]),
        ),
    ]
    for stack in stacks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ComplexImageFitWarning)
            fit = fit_complex_images(stack)
        assert np.isfinite(fit.rms), f"nan rms for {stack.rhos}"
        assert np.isfinite(fit.rms_extrapolated)
        assert fit.rms >= 0.0


def test_fit_warns_when_the_pencil_order_is_too_low() -> None:
    """An unreliable fit surfaces as a warning (F39).

    Pre-fix a structurally impossible fit was returned without any
    diagnostic; the only trace was a NaN or |K_1|-sized number in the
    result metadata.
    """
    stack = _three_layer_stack()
    with pytest.warns(ComplexImageFitWarning, match="unreliable fit"):
        fit = fit_complex_images(stack, n_images=1)
    assert not fit.converged
    assert fit.rms > 1e-3


def test_fit_warns_when_the_window_misses_the_decay_band() -> None:
    """The verification grid catches window truncation (F39).

    A stack whose relevant thicknesses span 0.05 m … 8 m cannot be
    covered by a single uniform window of 64 samples: the residual on
    the *fitting* grid is tiny (4.7e-06) while the residual on the
    independent log-spaced grid is 1.1e-02. Reporting only the former
    would look like success.
    """
    stack = LayerStack(
        rhos=np.array([100.0, 4000.0, 20.0, 900.0, 60.0]),
        h=np.array([0.2, 0.05, 8.0, 0.05]),
    )
    with pytest.warns(ComplexImageFitWarning, match="verification grid"):
        fit = fit_complex_images(stack)
    assert fit.rms < 1e-3 < fit.rms_extrapolated
    assert not fit.converged


def test_fit_warning_can_be_suppressed_without_losing_the_flag() -> None:
    """``warn=False`` keeps the diagnostics but silences the warning."""
    stack = _three_layer_stack()
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        fit = fit_complex_images(stack, n_images=1, warn=False)
    assert not any(
        issubclass(w.category, ComplexImageFitWarning) for w in recorded
    )
    assert not fit.converged


def test_fit_grid_step_resolves_the_structure_scale() -> None:
    """Grid step from h_max, upper bound from h_min (F39).

    Pre-fix both bounds came from ``h_min``. For h = [0.5, 10] that
    gave Δ = 1.587 against the structure scale 1/(2·h_max) = 0.05, so
    the fit had a single usable sample in the transition band and P
    collapsed to 1. Post-fix the same stack is fitted to ~1e-13.
    """
    thin = LayerStack(
        rhos=np.array([100.0, 400.0, 50.0]), h=np.array([0.5, 10.0])
    )
    fit = fit_complex_images(thin)
    assert fit.a.size >= 3
    assert fit.rms < 1e-6
    assert fit.converged


# =====================================================================
# F34 — no complex-image fit on any solve path; honest metadata
# =====================================================================


def test_solve_cim_does_not_call_the_fit(monkeypatch) -> None:
    """``solve_cim`` must not evaluate ``fit_complex_images`` (F34).

    Pre-fix the call sat unconditionally in front of the
    discretisation, so poisoning the symbol made every ``cim`` solve
    fail.
    """
    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("fit_complex_images must not be called")

    monkeypatch.setattr(cim_mod, "fit_complex_images", _boom)
    monkeypatch.setattr(
        bem_mod, "fit_complex_images", _boom, raising=False
    )

    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    res = gf.create_engine(backend="cim", segment_length=SEG).solve(
        _rod_world(soil)
    )
    assert res.backend == "cim"


def test_solve_bem_does_not_call_the_fit(monkeypatch) -> None:
    """``solve_bem`` must not evaluate ``fit_complex_images`` (F34)."""
    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("fit_complex_images must not be called")

    monkeypatch.setattr(cim_mod, "fit_complex_images", _boom)
    monkeypatch.setattr(
        bem_mod, "fit_complex_images", _boom, raising=False
    )

    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    res = gf.create_engine(backend="bem", segment_length=SEG).solve(
        _ring_world(soil)
    )
    assert res.backend == "bem"


@pytest.mark.parametrize(
    "soil_kind,expected",
    [("homogeneous", "image"), ("two_layer", "image_2layer")],
)
def test_cim_metadata_is_honest(soil_kind: str, expected: str) -> None:
    """No NaN fit diagnostics; explicit ``cim_fit_used`` (F34).

    Pre-fix: ``{'cim_n_images': 0, 'cim_rms': nan}`` and no statement
    that the numbers are bit-identical to another backend.
    """
    soil = (
        gf.HomogeneousSoil(resistivity=100.0)
        if soil_kind == "homogeneous"
        else gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    )
    res = gf.create_engine(backend="cim", segment_length=SEG).solve(
        _rod_world(soil)
    )
    assert res.metadata["cim_fit_used"] is False
    assert res.metadata["cim_rms"] is None
    assert res.metadata["cim_n_images"] == 0
    assert res.metadata["reduces_to"] == expected


def test_bem_metadata_is_honest() -> None:
    """``bem`` reports that it reproduces ``mom`` (F34)."""
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)
    res = gf.create_engine(backend="bem", segment_length=SEG).solve(
        _ring_world(soil)
    )
    assert res.metadata["cim_fit_used"] is False
    assert res.metadata["cim_rms"] is None
    assert res.metadata["cim_n_images"] == 0
    assert res.metadata["reduces_to"] == "mom"


def test_reduces_to_metadata_is_true() -> None:
    """The ``reduces_to`` claim must actually hold (F34).

    ``cim`` == ``image_2layer`` (68.6735 Ω class of numbers) and
    ``bem`` == ``mom`` to floating-point associativity; this is the
    reason neither may be counted as an independent engine.
    """
    soil = gf.TwoLayerSoil(rho_1=100.0, rho_2=400.0, h_1=2.0)

    z_cim = (
        gf.create_engine(backend="cim", segment_length=SEG)
        .solve(_rod_world(soil))
        .cluster_impedance("g1")[0]
    )
    z_il2 = (
        gf.create_engine(backend="image_2layer", segment_length=SEG)
        .solve(_rod_world(soil))
        .cluster_impedance("g1")[0]
    )
    assert z_cim.real == pytest.approx(z_il2.real, rel=1e-12)

    z_bem = (
        gf.create_engine(backend="bem", segment_length=SEG)
        .solve(_ring_world(soil))
        .cluster_impedance("g1")[0]
    )
    z_mom = (
        gf.create_engine(backend="mom", segment_length=SEG)
        .solve(_ring_world(soil))
        .cluster_impedance("g1")[0]
    )
    assert z_bem.real == pytest.approx(z_mom.real, rel=1e-12)


def test_dead_complex_image_code_is_gone() -> None:
    """The unreachable n ≥ 3 helpers were deleted (F34).

    ``_cim_field_potential`` had no caller anywhere in ``src/``,
    ``tests/`` or ``docs/``; the ``fit`` argument threaded through
    ``_cim_self_kernel_factory`` / ``_build_Z_collocation`` only fed
    branches that ``solve_cim`` / ``solve_bem`` reject up front, and
    ``n_images`` / ``n_samples`` on ``solve_cim`` / ``solve_bem`` were
    unreachable through ``Engine.solve``.
    """
    assert not hasattr(cim_mod, "_cim_field_potential")

    assert "fit" not in inspect.signature(
        cim_mod._cim_self_kernel_factory
    ).parameters
    assert "fit" not in inspect.signature(
        bem_mod._build_Z_collocation
    ).parameters

    for fn in (cim_mod.solve_cim, bem_mod.solve_bem):
        params = inspect.signature(fn).parameters
        assert "n_images" not in params
        assert "n_samples" not in params


def test_n3_kernel_builders_raise_instead_of_silently_computing() -> None:
    """The remaining n ≥ 3 branches raise (F34).

    Pre-fix both builders happily produced a kernel from an incomplete
    Green's function (single image family), reachable by any future
    caller that skipped the front-door check.
    """
    stack = _three_layer_stack()
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        cim_mod._cim_self_kernel_factory(stack)

    pts = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    lengths = np.array([0.5, 0.5])
    radii = np.array([0.005, 0.005])
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        bem_mod._build_Z_collocation(pts, lengths, radii, stack)


@pytest.mark.parametrize("backend", ["cim", "bem"])
def test_three_layer_still_rejected(backend: str) -> None:
    """n ≥ 3 remains a loud error and points at the working engines."""
    soil = gf.MultiLayerSoil(
        layers=[
            gf.SoilLayer(resistivity=100.0, thickness=2.0),
            gf.SoilLayer(resistivity=400.0, thickness=3.0),
            gf.SoilLayer(resistivity=50.0, thickness=None),
        ]
    )
    eng = gf.create_engine(backend=backend, segment_length=SEG)
    with pytest.raises(NotImplementedError, match="mom_sommerfeld"):
        eng.solve(_rod_world(soil))


# =====================================================================
# F17 — the documentation must describe the code
# =====================================================================


def _read(rel: str) -> str:
    return (DOCS / rel).read_text(encoding="utf-8")


def test_engines_index_routes_multilayer_to_mom_sommerfeld() -> None:
    """The n ≥ 3 decision-tree branch must not point at cim/bem (F17).

    Pre-fix (``docs/engines/index.md:62-65``):
    ``├─ primary → cim`` and
    ``├─ alternative weighting→ bem``, although both raise
    ``NotImplementedError`` for n ≥ 3 since 0.11.0.
    """
    text = _read("engines/index.md")
    block = text.split("n_layers ≥ 3")[1].split("```")[0]

    assert "mom_sommerfeld" in block
    assert re.search(r"primary\s*→\s*mom_sommerfeld", block)
    assert not re.search(r"→\s*cim\b", block)
    assert not re.search(r"→\s*bem\b", block)
    assert "NotImplementedError" in block

    # The n = 2 branch must not sell bem as an alternative kernel.
    two_layer = text.split("n_layers = 2")[1].split("n_layers ≥ 3")[0]
    assert not re.search(r"→\s*bem\b", two_layer)

    # And the family map must warn about the two alias backends.
    assert "reduces_to" in text
    assert "not count them twice" in text or "aliases" in text


def test_cim_page_no_longer_advertises_multilayer() -> None:
    """``docs/engines/cim.md`` must match the code (F17/F34/F39)."""
    text = _read("engines/cim.md")

    # Stale claims (pre-fix cim.md:125, :141, :146, :150-151, :174).
    assert "the genuine matrix-pencil fit runs" not in text
    assert "tested up to $n = 8$" not in text
    assert (
        "| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, or "
        "`MultiLayerSoil` |"
    ) not in text
    assert "it covers the $n \\ge 3$ regime" not in text

    # Honest replacements.
    assert "NotImplementedError" in text
    assert "not an independent cross-check" in text
    assert "cim_fit_used" in text
    assert "ComplexImageFitWarning" in text
    # The asymptote story is documented, not just fixed in code.
    assert "K_1" in text and "asymptote" in text

    # The worked example must not be a 3-layer cim solve.
    example = text.split("## Example")[1]
    first_block = example.split("```python")[1].split("```")[0]
    assert "MultiLayerSoil" not in first_block


def test_bem_page_no_longer_advertises_multilayer() -> None:
    """``docs/engines/bem.md`` must match the code (F17/F34)."""
    text = _read("engines/bem.md")

    assert "($n \\ge 3$) → homogeneous self-kernel" not in text
    assert (
        "| Soil model | `HomogeneousSoil`, `TwoLayerSoil`, "
        "`MultiLayerSoil` |"
    ) not in text
    assert "| `cim` ($n \\ge 3$) | $\\le 5\\,\\%$ |" not in text

    assert "NotImplementedError" in text
    assert "not an independent cross-check" in text
    assert "reduces_to" in text


def test_concepts_backend_table_matches_the_code() -> None:
    """The eight-backend table must not claim ``any layered`` (F17)."""
    text = _read("concepts.md")
    rows = {
        m.group(1): m.group(2).strip()
        for m in re.finditer(
            r"^\|\s*`(\w+)`\s*\|([^|]*)\|", text, flags=re.MULTILINE
        )
    }
    assert "cim" in rows and "bem" in rows
    for backend in ("cim", "bem"):
        assert rows[backend] != "any layered", (
            f"concepts.md still lists {backend} as 'any layered'"
        )
        assert "2-layer" in rows[backend]
    assert "mom_sommerfeld" in rows

    assert "only engine that" in text or "only $n \\ge 3$" in text
    assert "not independent checks" in text


def test_adr_0002_selection_heuristic_is_amended() -> None:
    """ADR-0002 must carry the pass-9 amendment (F17/F34/F39)."""
    text = _read("adr/0002-engine-family.md")
    assert "Amendment 2026-07-30" in text
    # The superseded 2026-05 heuristic must be marked as such.
    assert "Superseded" in text
    assert "cim_fit_used" in text
    assert "ComplexImageFitWarning" in text
    # And the n >= 3 recommendation must now be mom_sommerfeld.
    in_force = text.split("Superseded")[1].split("## Mathematical")[0]
    assert "mom_sommerfeld" in in_force
    assert "NotImplementedError" in in_force
