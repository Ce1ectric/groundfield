"""The texts of the line kernel describe the kernel as it is (audits W3A-03, W3A-04).

After user decision D2 the near/far switch is blended, the far Tagg/Sunde
images carry the quadrupole term (or the point form, ``Engine.image_far_field``)
and image pairs use the transverse distance (audit W3A-02); a straight strip
in two-layer soil converges within a few 1e-6, not monotonically. Auditor
W3A found the units page and the ``Engine.kernel`` docstring still describing
the one-step switch and point-form far images, and the strip claimed
"within 7.7e-7, monotonically" on five sampled segment lengths. The pages of
the coordinator (known limitations, changelog) are checked there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.finding("10.28")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


# audit W3A-04
def test_w3a_04_no_text_describes_the_one_step_switch_as_current() -> None:
    units = _flat((ROOT / "docs/reference/units-and-conventions.md").read_text("utf-8"))
    assert "the results move wherever the" not in units
    assert "switches from its exact integral to its far form at a midpoint" not in units
    from groundfield.solver.engine import Engine

    doc = _flat(Engine.__doc__ or "")
    for stale in ("deviates by up to 0.05 %", "stay in point", "2.7 times as long"):
        assert stale not in doc, stale
    for path in (
        "src/groundfield/solver/_line_kernel.py",
        "src/groundfield/solver/_kernels.py",
        "src/groundfield/solver/image_2layer.py",
    ):
        text = _flat((ROOT / path).read_text("utf-8"))
        assert "they use the point form, with the exact Galerkin value" not in text
        assert "so the point-source approximation is used for them" not in text
        assert "images (``n >= 1``) as point sources — they are" not in text
    kernels = _flat((ROOT / "docs/explanation/kernels.md").read_text("utf-8"))
    assert "took 2.7 times as long with numba" not in kernels


# audit W3A-03
def test_w3a_03_the_strip_texts_do_not_claim_monotone_convergence() -> None:
    """The strip converges within 3.6e-6 on a fine grid of segment lengths,
    not monotonically (``tests/test_two_layer_far_images.py``)."""
    for path in (
        "docs/explanation/kernels.md",
        "docs/explanation/engines/image_2layer.md",
        "src/groundfield/solver/engine.py",
    ):
        text = _flat((ROOT / path).read_text("utf-8"))
        for sentence in re.split(r"(?<=[.;])\s", text):
            if "strip" in sentence and ("7.7e-7" in sentence or "monoton" in sentence):
                if "not monoton" in sentence:
                    continue
                pytest.fail(f"{path}: {sentence[:160]}")


#: The pages that describe ``Engine.image_far_field`` (stream fw3r2a's files).
_POINT_FORM_TEXTS = (
    "src/groundfield/solver/engine.py",
    "docs/explanation/performance.md",
    "docs/explanation/engines/image_2layer.md",
    "docs/explanation/kernels.md",
    "docs/reference/api/solver.md",
    "docs/adr/0015-line-source-kernel-default.md",
)


# audit W3R2A-04
def test_w3r2a_04_the_point_form_texts_give_the_measured_range() -> None:
    """``"point"`` against the default: networks up to 4.3e-4 (16 house grids
    0.4 m above a 1000/10 interface), not "2e-5 … 1.8e-4"; and it is not the
    result before wave 3 (up to 2.3e-4 from the continuous switch, D2)."""
    for path in _POINT_FORM_TEXTS:
        text = _flat((ROOT / path).read_text("utf-8"))
        for stale in ("2e-5 … 1.8e-4", "2 · 10⁻⁵ … 1.8 · 10⁻⁴", "1.9e-5 … 1.8e-4"):
            assert stale not in text, (path, stale)
        for sentence in re.split(r"(?<=[.;])\s", text):
            if "as before wave 3" in sentence and "point" in sentence:
                pytest.fail(f"{path}: {sentence[:160]}")
    for path in (
        "src/groundfield/solver/engine.py",
        "docs/explanation/engines/image_2layer.md",
    ):
        text = _flat((ROOT / path).read_text("utf-8"))
        assert "4.3e-4" in text and "2.3e-4" in text, path
    api = _flat((ROOT / "docs/reference/api/solver.md").read_text("utf-8"))
    assert "4.3 · 10⁻⁴" in api


# audit W3R2A-05, W3R2A-06
def test_w3r2a_05_the_transverse_distance_is_exact_for_long_wires_only() -> None:
    """The axis value is the circumference average of a long line; a finite
    one has an end effect of order a²/(d ℓ). Direct pairs of different wires
    keep the reduced kernel (a convention, -3.5e-4 for two strips 1 cm above
    a 1000/30 interface, 5 cm apart)."""
    text = _flat((ROOT / "src/groundfield/solver/_line_kernel.py").read_text("utf-8"))
    assert "exact for a parallel image at any distance" not in text
    assert "a^2/(d" in text and "+3.0e-4" in text
    assert "-3.5e-4" in text


# audit W3R2A-07
def test_w3r2a_07_the_golden_declaration_quotes_the_measured_move() -> None:
    declarations = ROOT / "tests/golden/expected_changes/released/2026.11.0"
    text = (declarations / "w3xl.toml").read_text("utf-8")
    assert "line_img2_ring_rods_within_layer 3.19e-7" in text
    assert "line_img2_ring_rods_within_layer 3.4e-7" not in text
