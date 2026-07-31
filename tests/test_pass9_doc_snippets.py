"""Executable-documentation harness — review pass 9, finding F27.

Nothing in this repository used to execute the fenced ``python`` code
blocks under ``docs/``, so six copy-pasteable snippets had silently
rotted against the API they document (wrong constructor signature,
a backend / soil-model combination the solver explicitly rejects, a
keyword argument that no longer exists).  This module closes that
gap: it extracts every ``python`` block from ``docs/**/*.md`` and
executes it, so a broken snippet fails CI.

Physical / methodological context
--------------------------------
The documentation is part of the model's validity statement: the
snippets fix which soil model each engine accepts (``image_2layer``
requires a :class:`~groundfield.soil.TwoLayerSoil`; ``cim`` and the
real image series stop at :math:`n \\le 2` layers), which mesh a
result was converged at, and how the stochastic generators are
seeded.  A snippet that cannot run is therefore not a cosmetic
defect — it documents a physical configuration the solver refuses,
which is exactly the kind of statement a reader trusts.

Design
------
**Per-page namespace, blocks in document order.**  Documentation
pages are narratives: block *k* routinely reuses ``world``,
``result`` or ``layout`` built in block *k-1* (see
``docs/examples/09_plot_gallery.md``, where one reference solve
feeds eight plot cells).  Each page is therefore one test that
executes its blocks sequentially into a single namespace.  A failure
is reported with the page path and the *starting line* of the
offending block, so the traceback points at the source line in the
Markdown file.

**Opt-out marker.**  Not every block is a runnable program; ADRs in
particular contain schema and signature sketches.  Such a block is
excluded by an HTML comment on the line immediately above its fence
(blank lines in between are allowed)::

    <!-- skip-doctest: schema sketch, not runnable code -->
    ```python
    inductance_model: Literal[None, "neumann"]   # default None
    ```

A whole page opts out with ``<!-- skip-doctest-page: reason -->``
anywhere in the file.  Both markers are HTML comments, so they are
invisible in the rendered MkDocs output and cost the reader nothing.
Always give a reason — the harness prints it in the skip report.
The marker is documented for docs authors in
``docs/installation.md`` ("Executable documentation").

**Network.**  Overpass / ``geo`` snippets must be marked, not tried:
the CI sandbox has no route to ``overpass-api.de``, and an
unmarked network block would fail (or hang) instead of skipping.

**Runtime.**  Plots are rendered on the ``Agg`` backend,
``plt.show()`` is neutralised, every page runs in a throwaway
working directory (snippets that write CSV/VTK/JSON must not litter
the checkout), and all figures are closed after each page.  Pages
whose snippets genuinely take minutes are listed in
:data:`SLOW_PAGES` and are skipped unless ``GF_DOC_SNIPPETS_SLOW=1``
is set; the whole default run is a handful of seconds per page.

**Known-broken blocks.**  :data:`EXPECTED_FAILURES` lists blocks
that are known to raise and are owned elsewhere (see the per-entry
comments).  They are treated as non-strict expected failures: the
harness records the exception and continues with the next block, and
an entry that starts passing does *not* fail the suite — it is
merely reported so the entry can be deleted.  Every entry is meant
to be removed, not to grow.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``use``)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

#: Fence languages treated as executable Python.
_PY_LANGS = frozenset({"python", "py", "python3"})

_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<ticks>`{3,}|~{3,})(?P<info>.*)$")
_SKIP_BLOCK_RE = re.compile(
    r"^\s*<!--\s*skip-doctest\s*(?::\s*(?P<reason>.*?))?\s*-->\s*$"
)
_SKIP_PAGE_RE = re.compile(
    r"<!--\s*skip-doctest-page\s*(?::\s*(?P<reason>.*?))?\s*-->"
)

#: Pages whose snippets take minutes rather than seconds.  Skipped
#: unless ``GF_DOC_SNIPPETS_SLOW=1``.  Keep this empty if you can:
#: a snippet that is too slow for CI is usually also too slow for the
#: reader who copy-pastes it, and trimming the demo (coarser
#: ``segment_length``, fewer realisations) is the better fix.
SLOW_PAGES: dict[str, str] = {}

#: Blocks that are known to raise and whose Markdown file is *not*
#: owned by this harness.  Key: page path relative to the repository
#: root.  Value: ``{block_index: reason}`` with the 0-based index of
#: the executable block on that page (skipped blocks are *not*
#: counted).  Non-strict: a fixed entry is reported, not failed.
#:
#: Every entry below was measured on 2026-07-30 (v0.14.1 + in-flight
#: 0.15.0 work) and handed to the group that owns the file; delete
#: the entry together with the fix.
EXPECTED_FAILURES: dict[str, dict[int, str]] = {
    # Fragment: uses an undefined ``world``; needs a skip-doctest
    # marker or a two-line preamble.  Owner: coupling docs group.
    "docs/api/coupling.md": {
        0: "NameError: 'world' is not defined (illustrative fragment)",
    },
    # Block 0: ``WorldGenerator()`` is called without its required
    # ``cfg`` argument (same defect as docs/examples/06_tn_model.md,
    # fixed there).  Block 1: live Overpass query.
    # Owner: generators docs group.
    "docs/api/generators.md": {
        0: "TypeError: WorldGenerator.__init__() missing 'cfg'",
        1: "requests to overpass-api.de (needs a skip-doctest marker)",
    },
    # Fragments referencing undefined ``result`` / ``world`` / ``fit``,
    # a literal ``...`` placeholder and a non-existent
    # 'my-bustype.json'.  Owner: io docs group.
    "docs/api/io.md": {
        0: "NameError: 'result' is not defined (illustrative fragment)",
        1: "NameError: 'world' is not defined (illustrative fragment)",
        2: "AttributeError on the literal '...' placeholder",
        3: "NameError: 'fit' is not defined (illustrative fragment)",
        4: "FileNotFoundError: 'my-bustype.json' (illustrative fragment)",
    },
    # backend='image_2layer' on a HomogeneousSoil world — the solver
    # rejects the combination (see docs/examples/02_engine_comparison.md
    # for the correct degenerate-two-layer form).
    # Owner: postprocess docs group.
    "docs/api/postprocess.md": {
        4: "TypeError: Backend 'image_2layer' requires TwoLayerSoil",
    },
    # ``from groundfield.postprocess import plot_geometry`` — the
    # helper is not exported under that name.
    # Owner: plot-gallery docs group.
    "docs/examples/09_plot_gallery.md": {
        7: "ImportError: cannot import name 'plot_geometry'",
    },
}


@dataclass(frozen=True)
class Snippet:
    """One fenced ``python`` block found in a Markdown page.

    Attributes
    ----------
    page : str
        Page path relative to the repository root.
    index : int
        0-based index among the *executable* blocks of the page.
    start_line : int
        1-based line number of the first source line (i.e. the line
        after the opening fence), used in tracebacks.
    source : str
        Dedented block body.
    """

    page: str
    index: int
    start_line: int
    source: str


@dataclass
class PageSnippets:
    """All ``python`` blocks of one page, plus its skip bookkeeping."""

    page: str
    snippets: list[Snippet] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)
    page_skip_reason: str | None = None


def _preceding_skip_reason(lines: list[str], fence_index: int) -> str | None:
    """Return the ``skip-doctest`` reason for the fence at ``fence_index``.

    Scans upwards over blank lines only, so the marker must be the
    last non-empty line before the fence.  Returns ``None`` when the
    block is not marked and ``""`` when it is marked without a reason.
    """
    i = fence_index - 1
    while i >= 0 and not lines[i].strip():
        i -= 1
    if i < 0:
        return None
    match = _SKIP_BLOCK_RE.match(lines[i])
    if match is None:
        return None
    return (match.group("reason") or "").strip()


def _fence_language(info: str) -> str:
    """Extract the language token from a fence info string.

    Handles ``python``, ``py``, ``python title="x"``, ``{.python}``
    and ``{ .python .annotate }`` alike.
    """
    token = info.strip().lstrip("{").strip()
    token = token.split()[0] if token else ""
    return token.lstrip(".").split(",")[0].strip('"').lower()


def parse_page(path: Path) -> PageSnippets:
    """Extract the executable ``python`` blocks from one Markdown page.

    Fences are matched with their own delimiter run and indentation,
    so a ``python`` block nested inside an outer (e.g. ``markdown``)
    fence is not mistaken for a top-level one.
    """
    rel = path.relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    out = PageSnippets(page=rel)

    page_match = _SKIP_PAGE_RE.search(text)
    if page_match is not None:
        out.page_skip_reason = (page_match.group("reason") or "").strip()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent, ticks, info = (
            match.group("indent"),
            match.group("ticks"),
            match.group("info"),
        )
        # Find the matching closing fence (same character, at least as
        # long, empty info string).
        close = i + 1
        while close < len(lines):
            candidate = _FENCE_RE.match(lines[close])
            if (
                candidate is not None
                and candidate.group("ticks")[0] == ticks[0]
                and len(candidate.group("ticks")) >= len(ticks)
                and not candidate.group("info").strip()
            ):
                break
            close += 1
        body = lines[i + 1 : close]
        if _fence_language(info) in _PY_LANGS:
            reason = _preceding_skip_reason(lines, i)
            if reason is None:
                source = "\n".join(
                    line[len(indent) :] if line.startswith(indent) else line
                    for line in body
                )
                out.snippets.append(
                    Snippet(
                        page=rel,
                        index=len(out.snippets),
                        start_line=i + 2,
                        source=source,
                    )
                )
            else:
                out.skipped.append((i + 1, reason or "no reason given"))
        i = close + 1
    return out


def collect_pages() -> list[PageSnippets]:
    """Parse every Markdown page under ``docs/`` (sorted, stable)."""
    return [parse_page(p) for p in sorted(DOCS_DIR.rglob("*.md"))]


ALL_PAGES = collect_pages()
EXECUTABLE_PAGES = [p for p in ALL_PAGES if p.snippets]


@pytest.fixture()
def snippet_sandbox(tmp_path, monkeypatch):
    """Isolate snippet side effects: cwd, ``plt.show``, open figures."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    try:
        yield tmp_path
    finally:
        plt.close("all")


def _run_block(snippet: Snippet, namespace: dict[str, object]) -> None:
    """Execute one block, swallowing its stdout."""
    code = compile(snippet.source, f"{snippet.page}:{snippet.start_line}", "exec")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, namespace)  # noqa: S102 — executing docs is the point


@pytest.mark.parametrize(
    "page",
    EXECUTABLE_PAGES,
    ids=[p.page for p in EXECUTABLE_PAGES],
)
def test_doc_snippets_execute(page: PageSnippets, snippet_sandbox) -> None:
    """Every unmarked ``python`` block on the page must run.

    All blocks of a page share one namespace and run in document
    order, mirroring how a reader works through the page.
    """
    if page.page_skip_reason is not None:
        pytest.skip(f"page opt-out: {page.page_skip_reason or 'no reason given'}")
    if page.page in SLOW_PAGES and not os.environ.get("GF_DOC_SNIPPETS_SLOW"):
        pytest.skip(
            f"slow page ({SLOW_PAGES[page.page]}); "
            "set GF_DOC_SNIPPETS_SLOW=1 to run it"
        )

    expected = EXPECTED_FAILURES.get(page.page, {})
    namespace: dict[str, object] = {"__name__": "__doc_snippet__"}
    notes: list[str] = []
    for snippet in page.snippets:
        started = time.perf_counter()
        try:
            _run_block(snippet, namespace)
        except BaseException as exc:  # noqa: BLE001 — report, don't mask
            if snippet.index in expected:
                notes.append(
                    f"expected failure honoured: block {snippet.index} "
                    f"(line {snippet.start_line}): {type(exc).__name__}"
                )
                continue
            raise AssertionError(
                f"documentation snippet failed: {page.page}"
                f":{snippet.start_line} (block {snippet.index})\n"
                f"{type(exc).__name__}: {exc}\n"
                "Fix the snippet, or mark it with "
                "'<!-- skip-doctest: reason -->' if it is an "
                "illustrative fragment."
            ) from exc
        else:
            if snippet.index in expected:
                notes.append(
                    f"STALE EXPECTED_FAILURES entry: {page.page} block "
                    f"{snippet.index} (line {snippet.start_line}) now "
                    "passes — delete the entry."
                )
        finally:
            elapsed = time.perf_counter() - started
            # Diagnostic only (never a failure): 30 s is well above the
            # slowest legitimate block, so a hit means the snippet grew
            # a full-scale solve.
            if elapsed > 30.0:
                notes.append(
                    f"slow block {snippet.index} (line "
                    f"{snippet.start_line}): {elapsed:.1f} s — consider "
                    "trimming it or listing the page in SLOW_PAGES"
                )
    for note in notes:
        print(f"[doc-snippets] {page.page}: {note}")


def test_collection_is_healthy() -> None:
    """Guard against a silently vacuous harness.

    A regex regression that collects zero blocks would make every
    page test pass, so pin the order of magnitude of what must be
    found: on 2026-07-30 the 54 Markdown pages under ``docs/`` hold
    72 ``python`` blocks, 15 of them marked ``skip-doctest``, leaving
    57 blocks on 29 pages that are executed here (10 of those are
    listed in :data:`EXPECTED_FAILURES`).
    """
    n_blocks = sum(len(p.snippets) for p in ALL_PAGES)
    n_skipped = sum(len(p.skipped) for p in ALL_PAGES)
    assert len(EXECUTABLE_PAGES) >= 25, EXECUTABLE_PAGES
    assert n_blocks >= 50, n_blocks
    assert n_skipped >= 10, n_skipped
    # Skips must always carry a reason.
    for page in ALL_PAGES:
        for line, reason in page.skipped:
            assert reason != "no reason given", f"{page.page}:{line}"


def test_expected_failure_registry_is_wellformed() -> None:
    """Registry entries must reference real pages and real blocks.

    Keeps the hand-off list from rotting into a set of stale keys
    that silently stop protecting anything.
    """
    by_page = {p.page: p for p in ALL_PAGES}
    for rel, blocks in EXPECTED_FAILURES.items():
        assert rel in by_page, f"unknown page in EXPECTED_FAILURES: {rel}"
        n = len(by_page[rel].snippets)
        for index in blocks:
            assert 0 <= index < n, f"{rel}: block {index} of {n}"
    for rel in SLOW_PAGES:
        assert rel in by_page, f"unknown page in SLOW_PAGES: {rel}"


# --------------------------------------------------------------------
# Focused regressions for the snippets fixed with F27.  These assert
# the *content* of the fix, not just that the page runs, so the intent
# survives a future rewrite of the page.
# --------------------------------------------------------------------


def _page_text(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_engine_comparison_pairs_image_2layer_with_two_layer_soil() -> None:
    """``image_2layer`` must never be offered a homogeneous soil.

    The engine raises ``TypeError: Backend 'image_2layer' requires
    TwoLayerSoil``; the physically homogeneous limit is reached from
    the two-layer side with :math:`\\rho_2 = \\rho_1` (reflection
    factor :math:`K = 0`), which is what the page now shows.
    """
    text = _page_text("docs/examples/02_engine_comparison.md")
    assert 'backends = ["image", "image_2layer", "mom", "bem", "fem"]' not in text
    assert "gf.TwoLayerSoil(rho_1=100.0, rho_2=100.0" in text
    assert "image_2layer` accepts a homogeneous soil too" not in text


def test_multilayer_page_uses_soillayer_and_a_backend_that_supports_n3() -> None:
    """``MultiLayerSoil`` takes ``layers``, and ``cim`` stops at n = 2.

    ``MultiLayerSoil(layer_resistivities=..., layer_thicknesses=...)``
    never existed as a signature, and ``cim`` rejects
    :math:`n \\ge 3` with ``NotImplementedError`` (the historic
    complex-image expansion of :math:`\\Gamma_1(\\lambda)` was
    structurally incomplete), so the n = 3 example must use
    ``mom_sommerfeld``.
    """
    text = _page_text("docs/examples/03_multilayer_soil.md")
    assert "layer_resistivities=" not in text
    assert "gf.SoilLayer(resistivity=80.0, thickness=3.0)" in text
    assert 'backend="mom_sommerfeld"' in text
    assert 'backend="cim"' not in text


def test_tn_model_page_uses_the_real_generator_signature() -> None:
    """``TnNetworkGenerator(cfg, seed=...)`` — cfg is bound at init.

    ``build()`` takes only an optional config *override* and no
    ``rng`` argument; the RNG lives on the generator so that one
    seed reproduces the whole stochastic realisation.
    """
    text = _page_text("docs/examples/06_tn_model.md")
    assert "gf.TnNetworkGenerator(cfg, seed=42)" in text
    assert "gen.build(cfg, rng=" not in text
    assert "gf.np.random" not in text


def test_network_and_sketch_blocks_are_marked() -> None:
    """Overpass-dependent and sketch-only blocks carry the marker.

    Overpass is unreachable from CI (and rate-limited everywhere
    else), so those blocks must opt out explicitly rather than fail.
    """
    osm = parse_page(REPO_ROOT / "docs/examples/08_osm_pipeline.md")
    assert osm.snippets == []
    assert len(osm.skipped) == 2
    perf = parse_page(REPO_ROOT / "docs/performance.md")
    assert perf.snippets == []
    for adr in ("0004-inductive-coupling", "0005-carson-earth-return"):
        page = parse_page(REPO_ROOT / f"docs/adr/{adr}.md")
        assert page.snippets == [], adr
