"""Content checks of the explanation pages and ADRs (docs findings 10.83, 10.85, 10.91).

The explanation section replaces ``docs/engines/*``, ``docs/concepts.md``
and ``docs/performance.md``. These tests pin the statements the review
found wrong in the old pages (10.83) — so they cannot come back — and the
structural rules of the documentation plan: one theory page per engine
with a mathematical-physical context, no reST roles (10.85), dated ADR
amendments, links and anchors that resolve.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
EXPLANATION = DOCS / "explanation"

ENGINE_PAGES = ("image", "image_2layer", "mom", "mom_sommerfeld", "fem")
TOPIC_PAGES = (
    "kernels",
    "earth-return",
    "concrete-shells",
    "sources-and-measurement",
)
AMENDED_ADRS = (
    "0002-engine-family",
    "0004-inductive-coupling",
    "0005-carson-earth-return",
    "0006-sommerfeld-earth-return",
    "0007-cross-layer-electrodes",
    "0008-groundinsight-bridge",
    "0009-world-generators",
    "0012-foundation-concrete-encasement",
)
NEW_ADRS = (
    "0013-calendar-versioning",
    "0014-backend-registry-and-pipeline",
    "0015-line-source-kernel-default",
)
#: Link targets owned by other documentation streams; checked once they exist.
EXTERNAL_TARGETS = {"how-to/index.md", "tutorials/index.md"}


def _own_pages() -> list[Path]:
    """The explanation pages and every ADR."""
    return sorted(EXPLANATION.rglob("*.md")) + sorted((DOCS / "adr").glob("*.md"))


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _engine(name: str) -> str:
    return _text(EXPLANATION / "engines" / f"{name}.md")


def test_the_explanation_pages_of_the_plan_exist() -> None:
    expected = [EXPLANATION / f"{p}.md" for p in ("index", "concepts", *TOPIC_PAGES)]
    expected += [EXPLANATION / "validation.md", EXPLANATION / "performance.md"]
    expected += [EXPLANATION / "engines" / f"{p}.md" for p in ("index", *ENGINE_PAGES)]
    missing = [str(p.relative_to(REPO)) for p in expected if not p.exists()]
    assert missing == []


def test_the_replaced_pages_are_gone() -> None:
    assert not (DOCS / "engines").exists()
    assert not (DOCS / "concepts.md").exists()
    assert not (DOCS / "performance.md").exists()


@pytest.mark.parametrize("name", ENGINE_PAGES + ("index",))
def test_every_engine_page_states_the_mathematical_physical_context(
    name: str,
) -> None:
    """User rule: every engine page gives the physics, not only the API."""
    assert "## Mathematical and physical context" in _engine(name)


@pytest.mark.parametrize("name", TOPIC_PAGES)
def test_every_topic_page_states_the_mathematical_physical_context(
    name: str,
) -> None:
    assert "## Mathematical and physical context" in _text(EXPLANATION / f"{name}.md")


@pytest.mark.parametrize("name", ENGINE_PAGES)
def test_every_engine_page_has_validity_validation_and_literature(name: str) -> None:
    text = _engine(name)
    for heading in ("## Validity range", "## Validation", "## Literature"):
        assert heading in text, f"{name}: {heading} missing"
    assert f"<!-- generated:begin {name} -->" in text


@pytest.mark.finding("10.85")
@pytest.mark.parametrize("page", _own_pages(), ids=lambda p: p.name)
def test_no_rest_roles(page: Path) -> None:
    """Sphinx roles render as raw text in MkDocs (10.85)."""
    roles = re.findall(
        r":(?:class|func|meth|math|mod|attr|data|exc|obj):`", _text(page)
    )
    assert roles == []


@pytest.mark.finding("10.83")
def test_multi_layer_soils_are_routed_to_mom_sommerfeld() -> None:
    """F17: the old decision tree sent n >= 3 to cim / bem, which raise."""
    index = _engine("index")
    assert "| Three or more distinct layers | `mom_sommerfeld` |" in index
    for alias in ("cim", "bem", "image_nlayer"):
        assert f"**`{alias}`**" in index, f"{alias} not explained"
    assert "deprecated" in index and "2027.5" in index
    # The decision guide opens the page since the final docs pass of 2026.11;
    # its section runs up to the next level-2 heading.
    choosing = index.split("## Choosing an engine")[1].split("\n## ")[0]
    assert "### Decision path" in choosing
    assert not re.search(r"`(cim|bem|image_nlayer)`", choosing)


@pytest.mark.finding("10.83", "10.31")
def test_mom_is_described_as_point_matching_not_galerkin() -> None:
    text = _engine("mom")
    assert "point matching" in text
    assert "This is **not** a Galerkin scheme" in text
    for page in (*ENGINE_PAGES, "index"):
        body = _engine(page)
        assert "Galerkin MoM" not in body
        assert "Galerkin method-of-moments" not in body.lower()


@pytest.mark.finding("10.83", "10.23")
def test_fem_is_experimental_and_an_independent_reference() -> None:
    """Up to 2026.11 the page said "not as a cross-check" (the equivalent
    hemisphere, 10.23); the conforming model is the independent reference
    (user decision Q37): axisymmetric where it applies, three-dimensional
    for every other world (part B), and the page says which applies."""
    text = " ".join(_engine("fem").split())
    assert "experimental" in text.lower()
    assert "independent reference" in text
    assert "three-dimensional model" in text and 'model="axisymmetric"' in text
    assert "≤ 10 %" not in text and "<= 10 %" not in text


@pytest.mark.finding("10.83", "10.20")
def test_carson_series_is_not_sold_as_identical_to_sommerfeld() -> None:
    for page in ("earth-return", "performance", "concepts"):
        text = _text(EXPLANATION / f"{page}.md").lower()
        assert "identical impedances" not in text
        assert "1200" not in text
    assert "anti-parallel" in _text(EXPLANATION / "earth-return.md")


@pytest.mark.finding("10.83")
def test_perfect_mirror_is_not_called_the_default() -> None:
    for page in _own_pages():
        if page.parent.name == "adr":
            continue  # ADR bodies keep their historical text
        text = _text(page)
        assert not re.search(r"perfect_mirror[`\"]*\s*\(default", text), page.name


@pytest.mark.finding("10.83", "10.15")
def test_image_2layer_explains_electrodes_through_the_interface() -> None:
    text = _engine("image_2layer")
    assert "raises for z" not in text  # the 0.15.0 claim that such geometry raises
    assert "one owner per layer" in text


@pytest.mark.finding("10.18")
def test_the_three_impedances_are_defined() -> None:
    text = _text(EXPLANATION / "sources-and-measurement.md")
    assert "## Three impedances" in text
    for accessor in ("input_impedance", "cluster_impedance", "grounding_impedance"):
        assert accessor in text


@pytest.mark.parametrize("name", AMENDED_ADRS)
def test_amended_adrs_carry_a_dated_2026_11_amendment(name: str) -> None:
    text = _text(DOCS / "adr" / f"{name}.md")
    assert "## Amendment 2026.11" in text
    status = next(line for line in text.splitlines() if line.startswith("| **Status**"))
    assert "amended 2026.11" in status


@pytest.mark.parametrize("name", NEW_ADRS)
def test_new_adrs_are_accepted(name: str) -> None:
    text = _text(DOCS / "adr" / f"{name}.md")
    assert re.search(r"^# ADR-00(13|14|15): ", text, flags=re.MULTILINE)
    assert "| **Status** | Accepted |" in text
    for heading in ("## Context", "## Decision", "## Consequences"):
        assert heading in text


# ---------------------------------------------------------------------
# Links and anchors
# ---------------------------------------------------------------------

_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    """Python-Markdown's default ``toc`` slug of a heading."""
    text = re.sub(r"`|\*\*?|\$", "", heading)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _anchors(path: Path) -> set[str]:
    return {
        _slug(m.group(1))
        for m in re.finditer(r"^#{1,6}\s+(.+?)\s*$", _text(path), flags=re.MULTILINE)
    }


def _links(page: Path) -> list[tuple[Path, str, str]]:
    out = []
    for target in _LINK.findall(_text(page)):
        if re.match(r"[a-z]+:", target):
            continue  # external URL
        path, _, anchor = target.partition("#")
        resolved = (page.parent / path).resolve() if path else page
        out.append((resolved, anchor, target))
    return out


@pytest.mark.parametrize("page", _own_pages(), ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(page: Path) -> None:
    broken = []
    for resolved, anchor, target in _links(page):
        if not resolved.exists():
            rel = (
                resolved.relative_to(DOCS).as_posix()
                if DOCS in resolved.parents
                else ""
            )
            if rel not in EXTERNAL_TARGETS:
                broken.append(target)
            continue
        if anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(target)
    assert broken == []
