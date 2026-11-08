"""Structure of the changelog, its documentation page and the audit history.

Release 2026.11.0 condensed ``CHANGELOG.md`` into release notes and moved the
narratives of the review passes verbatim to ``docs/audit/`` (finding 10.93).
These tests keep the pieces consistent:

* the documentation page ``docs/changelog.md`` includes the section of
  ``CHANGELOG.md`` between the snippet markers, and both files define the
  reference-style links into the audit history (``CHANGELOG.md`` relative to
  the repository root for reading on GitHub, the page relative to ``docs/``);
* the ``[Unreleased]`` block carries the blocks every release must have
  (*Numbers that change*; a test for every *Fixed* entry);
* the 0.15.0 statements corrected by review pass 10 keep their footnotes
  (finding 10.82);
* the pass-10 page holds every changelog fragment verbatim while
  ``changelog.d/`` exists, so nothing of the work-stream notes is lost when
  the fragments are deleted at the release.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CHANGELOG = REPO / "CHANGELOG.md"
DOCS_PAGE = REPO / "docs" / "changelog.md"
AUDIT = REPO / "docs" / "audit"
FRAGMENTS = REPO / "changelog.d"

START = "<!-- --8<-- [start:changelog] -->"
END = "<!-- --8<-- [end:changelog] -->"

# A reference-style link ``[text][label]`` and a definition ``[label]: target``.
_USE_RE = re.compile(r"\]\[([a-z0-9-]+)\]")
_DEF_RE = re.compile(r"^\[([^\]^\n]+)\]:[ \t]+(\S+)", re.MULTILINE)
_HEADING_RE = re.compile(r"^## \[([^\]]+)\](.*)$", re.MULTILINE)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _unreleased() -> str:
    text = _text(CHANGELOG)
    body = text.split("\n## [Unreleased]\n", 1)[1]
    return body.split("\n## [", 1)[0]


def _section(version: str) -> str:
    text = _text(CHANGELOG)
    body = text.split(f"\n## [{version}] - ", 1)[1]
    return body.split("\n## [", 1)[0]


def _notes_2026_11() -> str:
    """Release notes of the first calendar release (planned as 2026.11.0).

    The ``[Unreleased]`` block until the release tool turns it into
    ``## [YYYY.M.MICRO] - <date>``; afterwards the oldest calendar-versioned
    section, whatever its number. The tests below therefore hold before and
    after the release commit, so the release workflow, which runs them on
    the tag, passes.
    """
    body = _unreleased()
    if "\n### " in body:
        return body
    calver = [
        v for v, _ in _HEADING_RE.findall(_text(CHANGELOG))
        if re.fullmatch(r"20\d\d\.\d+\.\d+", v)
    ]
    return _section(calver[-1])


@pytest.mark.finding("10.93")
def test_docs_page_includes_the_marked_section() -> None:
    text = _text(CHANGELOG)
    assert text.count(START) == 1 and text.count(END) == 1
    assert text.index(START) < text.index("## [Unreleased]") < text.index(END)
    assert '--8<-- "CHANGELOG.md:changelog"' in _text(DOCS_PAGE)


@pytest.mark.finding("10.93")
def test_audit_links_are_defined_for_github_and_for_the_docs() -> None:
    text = _text(CHANGELOG)
    inside = text.split(START, 1)[1].split(END, 1)[0]
    outside = text.split(END, 1)[1]
    used = set(_USE_RE.findall(inside))
    assert used, "the release notes link to the audit history"
    repo_defs = dict(_DEF_RE.findall(outside))
    docs_defs = dict(_DEF_RE.findall(_text(DOCS_PAGE)))
    assert used <= set(repo_defs), f"undefined in CHANGELOG.md: {used - set(repo_defs)}"
    assert used <= set(docs_defs), (
        f"undefined in docs/changelog.md: {used - set(docs_defs)}"
    )
    for label in used:
        assert (REPO / repo_defs[label]).is_file(), repo_defs[label]
        assert (REPO / "docs" / docs_defs[label]).is_file(), docs_defs[label]
    # the version links stay inside the included section, the audit links outside
    assert not set(dict(_DEF_RE.findall(inside))) & used


@pytest.mark.finding("10.93")
def test_release_headings_follow_keep_a_changelog() -> None:
    headings = _HEADING_RE.findall(_text(CHANGELOG))
    assert headings[0] == ("Unreleased", "")
    for version, rest in headings[1:]:
        assert re.fullmatch(r" - \d{4}-\d{2}-\d{2}", rest), f"## [{version}]{rest}"
    dates = [rest[3:] for _, rest in headings[1:]]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.finding("10.93")
def test_unreleased_block_has_the_mandatory_blocks() -> None:
    body = _notes_2026_11()
    for heading in (
        "### Numbers that change",
        "### Security",
        "### Deprecated",
        "### Removed",
        "### Fixed",
        "### Corrections to the 0.15.0 release notes",
    ):
        assert f"\n{heading}\n" in body, heading
    security = body.split("\n### Security\n", 1)[1]
    first_entry = security.split("\n- ", 1)[1]
    assert first_entry.startswith("**") and "(10.01)" in first_entry.split("\n", 1)[0]


@pytest.mark.finding("10.93")
def test_every_fixed_entry_names_its_test() -> None:
    fixed = _notes_2026_11().split("\n### Fixed\n", 1)[1].split("\n### ", 1)[0]
    entries = re.split(r"\n- ", "\n" + fixed)[1:]
    assert entries
    for entry in entries:
        assert "`tests/" in entry, entry.split("\n", 1)[0]


@pytest.mark.finding("10.93")
def test_release_tool_reads_the_whole_unreleased_block() -> None:
    """``scripts/release.py`` ends the block at the first link definition."""
    body = _unreleased()
    assert not _DEF_RE.search(body)
    assert "## [Unreleased]" not in _text(CHANGELOG).split("\n## [Unreleased]\n", 1)[0]


def test_no_display_math_and_no_local_paths() -> None:
    text = _text(CHANGELOG)
    assert "$$" not in text
    for pattern in ("/home/", "/Users/", "C:\\", "Nextcloud"):
        assert pattern not in text, pattern


@pytest.mark.finding("10.82")
def test_corrected_0150_statements_keep_their_footnotes() -> None:
    section = _section("0.15.0")
    for label in ("c-cim", "c-f16", "c-f17", "c-f47", "c-f48", "c-f53"):
        assert f"[^{label}]" in section.split(f"[^{label}]:", 1)[0], label
        assert f"\n[^{label}]: Correction (2026.11.0)" in section, label
    corrections = _notes_2026_11().split(
        "\n### Corrections to the 0.15.0 release notes\n", 1
    )[1]
    for finding in ("F47", "F48", "F53", "F54", "F16", "F08", "F17", "`cim`"):
        assert finding in corrections, finding


@pytest.mark.finding("10.93")
def test_audit_index_links_every_pass_page() -> None:
    pages = sorted(p.name for p in AUDIT.glob("*.md") if p.name != "index.md")
    assert {f"pass-{n:02d}.md" for n in range(1, 11)} <= set(pages)
    index = _text(AUDIT / "index.md")
    for page in pages:
        assert f"]({page})" in index, page


@pytest.mark.finding("10.93")
def test_pass10_page_holds_every_fragment_verbatim() -> None:
    fragments = sorted(p for p in FRAGMENTS.glob("*.md") if p.name != "README.md")
    if not fragments:
        pytest.skip("no changelog fragments (they are deleted at the release)")
    page = _text(AUDIT / "pass-10.md")
    stale = [p.name for p in fragments if _text(p).strip("\n") not in page]
    assert not stale, (
        "docs/audit/pass-10.md does not hold these fragments verbatim; regenerate "
        f"the page from changelog.d/ (see the docd report): {stale}"
    )
