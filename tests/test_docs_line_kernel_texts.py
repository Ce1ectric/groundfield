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
