"""Executable documentation: every Python block of the docs runs (10.88).

Every fenced ``python`` block in ``docs/**/*.md`` and in ``README.md`` is
executed, page by page, blocks in document order in one namespace (a page
is a narrative: block *k* reuses the objects of block *k-1*). A block
either runs, or it is marked explicitly as something else. Up to 2026.11
the harness tolerated ten raising blocks through a registry of expected
failures, checked neither values nor warnings and skipped the README
(review pass 10, finding 10.88); this module replaces it.

Markers
-------
All markers are HTML comments, invisible in the rendered site. A block
marker stands on the line(s) directly above the fence (blank lines in
between are allowed; several markers may be stacked):

``<!-- skip-doctest: reason -->``
    The block is a fragment (a schema sketch, a signature, a call that
    needs the network) and is not executed. The reason is mandatory.
``<!-- doctest-module: name.py -->``
    The block is the content of a module file: it is written to
    ``name.py`` in the page's working directory, which is on
    ``sys.path``, so that later blocks (and the worker processes of
    ``sweep(n_jobs=...)``) can import it.
``<!-- doctest-allow-warnings: CategoryA, CategoryB -->``
    The block may emit warnings of these categories (class names; a
    subclass matches). Every other warning that comes from groundfield
    or from the documented code fails the page (see *Warnings*).

A page opts out as a whole with ``<!-- skip-doctest-page: reason -->``
anywhere in the file.

Value checks
------------
A multi-line comment that starts with ``doctest-check``::

    <!-- doctest-check
    assert abs(z_in.real - 33.4997) < 1e-3
    -->

is executed at its position in the page, in the page's namespace. It
checks a number the page prints without cluttering the code the reader
copies. Pages under :data:`REQUIRE_VALUE_CHECKS` must carry at least one
``assert`` (in a block or in a check comment).

Warnings
--------
Inside every block all warnings are recorded. A warning *belongs to the
page* when its category is defined in groundfield, when it is raised from
groundfield's code, or when it is attributed to the documented code
itself (groundfield points its deprecation warnings at the caller's
line). Such a warning must be listed in the block's
``doctest-allow-warnings`` marker; in particular a page that uses a
deprecated name fails. Warnings of third-party libraries are reported,
not failed.

Runtime
-------
Plots use the ``Agg`` backend and ``plt.show()`` is neutralised; every
page runs in its own temporary working directory, which is prepended to
``sys.path``; figures are closed after each page. Pages listed in
:data:`SLOW_PAGES` run only with ``GF_DOC_SNIPPETS_SLOW=1``.

Pages of the old layout
-----------------------
:data:`LEGACY_PREFIXES` names the pages of the documentation layout
before 2026.11 (``docs/api``, ``docs/engines``, the old ``docs/examples``, the
old top-level pages). The 2026.11 restructure (Diátaxis) deletes or moves
all of them; while one still exists in a working tree, its blocks run
without strictness and failures are only reported. The new layout never
uses these paths, so the list protects nothing after the restructure and
can be deleted then.

The tutorials (``docs/tutorials/*.ipynb``) are executed by
``tests/test_tutorials.py``.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow ``use``)

import groundfield  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"
GROUNDFIELD_DIR = Path(groundfield.__file__).resolve().parent

#: Fence languages treated as executable Python.
_PY_LANGS = frozenset({"python", "py", "python3"})

#: Pages that must carry at least one value check (``assert``), as path
#: prefixes relative to the repository root.
REQUIRE_VALUE_CHECKS: tuple[str, ...] = ("docs/how-to/",)

#: Pages whose snippets take minutes rather than seconds. Skipped unless
#: ``GF_DOC_SNIPPETS_SLOW=1``. Keep it empty: a snippet that is too slow
#: for CI is too slow for the reader who copies it.
SLOW_PAGES: dict[str, str] = {}

#: Paths of the documentation layout before 2026.11 (see the module
#: docstring). Matched as prefixes of the page path.
LEGACY_PREFIXES: tuple[str, ...] = (
    "docs/api/",
    "docs/engines/",
    "docs/concepts.md",
    "docs/performance.md",
    "docs/installation.md",
    "docs/quickstart.md",
)

_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<ticks>`{3,}|~{3,})(?P<info>.*)$")
_SKIP_BLOCK_RE = re.compile(
    r"^\s*<!--\s*skip-doctest\s*(?::\s*(?P<reason>.*?))?\s*-->\s*$"
)
_MODULE_RE = re.compile(r"^\s*<!--\s*doctest-module\s*:\s*(?P<name>\S*)\s*-->\s*$")
_ALLOW_RE = re.compile(
    r"^\s*<!--\s*doctest-allow-warnings\s*:\s*(?P<names>.*?)\s*-->\s*$"
)
_SKIP_PAGE_RE = re.compile(r"<!--\s*skip-doctest-page\s*(?::\s*(?P<reason>.*?))?\s*-->")
_CHECK_OPEN_RE = re.compile(r"^\s*<!--\s*doctest-check\s*$")
_CHECK_CLOSE_RE = re.compile(r"^\s*-->\s*$")
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")


@dataclass(frozen=True)
class Snippet:
    """One unit of a page: a code block, a module file or a check.

    Attributes
    ----------
    page : str
        Page path relative to the repository root.
    kind : str
        ``"block"`` (executed), ``"module"`` (written to a file) or
        ``"check"`` (a ``doctest-check`` comment, executed).
    start_line : int
        1-based line number of the first source line, used in tracebacks.
    source : str
        Dedented source.
    allowed_warnings : tuple of str
        Warning category names the block may emit.
    module_name : str
        File name for ``kind == "module"``.
    """

    page: str
    kind: str
    start_line: int
    source: str
    allowed_warnings: tuple[str, ...] = ()
    module_name: str = ""


@dataclass
class Page:
    """All units of one page, plus its skip bookkeeping."""

    page: str
    units: list[Snippet] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)
    page_skip_reason: str | None = None
    marker_errors: list[str] = field(default_factory=list)

    @property
    def executable(self) -> list[Snippet]:
        return [u for u in self.units if u.kind in ("block", "check")]

    @property
    def is_legacy(self) -> bool:
        return self.page.startswith(LEGACY_PREFIXES)


def _fence_language(info: str) -> str:
    """Language token of a fence info string (``python title="x"`` etc.)."""
    token = info.strip().lstrip("{").strip()
    token = token.split()[0] if token else ""
    return token.lstrip(".").split(",")[0].strip('"').lower()


def _markers_above(lines: list[str], fence_index: int) -> list[str]:
    """Marker comment lines directly above a fence (blank lines skipped)."""
    markers: list[str] = []
    i = fence_index - 1
    while i >= 0:
        line = lines[i]
        if not line.strip():
            i -= 1
            continue
        if (
            _SKIP_BLOCK_RE.match(line)
            or _MODULE_RE.match(line)
            or _ALLOW_RE.match(line)
        ):
            markers.append(line)
            i -= 1
            continue
        break
    return markers


def parse_page(path: Path) -> Page:
    """Split one Markdown page into blocks, module files and checks."""
    rel = path.resolve().relative_to(REPO_ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    out = Page(page=rel)
    page_match = _SKIP_PAGE_RE.search(text)
    if page_match is not None:
        out.page_skip_reason = (page_match.group("reason") or "").strip()

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _CHECK_OPEN_RE.match(lines[i]):
            close = i + 1
            while close < len(lines) and not _CHECK_CLOSE_RE.match(lines[close]):
                close += 1
            if close >= len(lines):
                out.marker_errors.append(
                    f"line {i + 1}: doctest-check comment is never closed"
                )
                break
            out.units.append(
                Snippet(
                    page=rel,
                    kind="check",
                    start_line=i + 2,
                    source="\n".join(lines[i + 1 : close]),
                )
            )
            i = close + 1
            continue
        match = _FENCE_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent, ticks, info = (
            match.group("indent"),
            match.group("ticks"),
            match.group("info"),
        )
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
        if _fence_language(info) in _PY_LANGS:
            source = "\n".join(
                line[len(indent) :] if line.startswith(indent) else line
                for line in lines[i + 1 : close]
            )
            skip_reason, module_name, allowed = None, "", []
            for marker in _markers_above(lines, i):
                if (m := _SKIP_BLOCK_RE.match(marker)) is not None:
                    skip_reason = (m.group("reason") or "").strip()
                    if not skip_reason:
                        out.marker_errors.append(
                            f"line {i + 1}: skip-doctest marker without a reason"
                        )
                elif (m := _MODULE_RE.match(marker)) is not None:
                    module_name = m.group("name")
                    if not _MODULE_NAME_RE.match(module_name):
                        out.marker_errors.append(
                            f"line {i + 1}: invalid doctest-module name {module_name!r}"
                        )
                elif (m := _ALLOW_RE.match(marker)) is not None:
                    allowed.extend(
                        n.strip() for n in m.group("names").split(",") if n.strip()
                    )
            if skip_reason is not None:
                out.skipped.append((i + 1, skip_reason))
            elif module_name:
                out.units.append(
                    Snippet(
                        page=rel,
                        kind="module",
                        start_line=i + 2,
                        source=source,
                        module_name=module_name,
                    )
                )
            else:
                out.units.append(
                    Snippet(
                        page=rel,
                        kind="block",
                        start_line=i + 2,
                        source=source,
                        allowed_warnings=tuple(allowed),
                    )
                )
        i = close + 1
    return out


def collect_pages() -> list[Page]:
    """Parse every Markdown page under ``docs/`` and the README."""
    paths = sorted(DOCS_DIR.rglob("*.md"))
    readme = REPO_ROOT / "README.md"
    if readme.exists():
        paths.append(readme)
    return [parse_page(p) for p in paths]


ALL_PAGES = collect_pages()
EXECUTABLE_PAGES = [p for p in ALL_PAGES if p.executable]


def _belongs_to_page(message: warnings.WarningMessage, code_prefix: str) -> bool:
    """Whether a recorded warning is the page's responsibility."""
    category = message.category
    if category.__module__.split(".")[0] == "groundfield":
        return True
    filename = str(message.filename)
    if filename.startswith(code_prefix):
        return True
    try:
        Path(filename).resolve().relative_to(GROUNDFIELD_DIR)
    except (ValueError, OSError):
        return False
    return True


def _is_allowed(category: type[Warning], allowed: tuple[str, ...]) -> bool:
    return any(base.__name__ in allowed for base in category.__mro__)


def _run_unit(
    unit: Snippet, namespace: dict[str, object]
) -> list[warnings.WarningMessage]:
    """Execute one block or check; return the warnings it emitted."""
    code = compile(unit.source, f"{unit.page}:{unit.start_line}", "exec")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)  # noqa: S102 — executing the docs is the point
    return list(caught)


@pytest.fixture()
def snippet_sandbox(tmp_path, monkeypatch):
    """Isolated working directory on ``sys.path``; no plot windows."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    modules_before = set(sys.modules)
    try:
        yield tmp_path
    finally:
        plt.close("all")
        # Module files of one page must not leak into the next page.
        for name in set(sys.modules) - modules_before:
            module = sys.modules.get(name)
            if module is not None and str(
                getattr(module, "__file__", "") or ""
            ).startswith(str(tmp_path)):
                del sys.modules[name]


@pytest.mark.parametrize(
    "page", EXECUTABLE_PAGES, ids=[p.page for p in EXECUTABLE_PAGES]
)
def test_doc_snippets_execute(page: Page, snippet_sandbox: Path) -> None:
    """Every unmarked block runs and emits only allowed warnings."""
    if page.page_skip_reason is not None:
        pytest.skip(f"page opt-out: {page.page_skip_reason or 'no reason given'}")
    if page.page in SLOW_PAGES and not os.environ.get("GF_DOC_SNIPPETS_SLOW"):
        pytest.skip(
            f"slow page ({SLOW_PAGES[page.page]}); set GF_DOC_SNIPPETS_SLOW=1 to run it"
        )

    namespace: dict[str, object] = {"__name__": "__doc_snippet__"}
    notes: list[str] = []
    for unit in page.units:
        if unit.kind == "module":
            (snippet_sandbox / unit.module_name).write_text(
                unit.source + "\n", encoding="utf-8"
            )
            continue
        started = time.perf_counter()
        where = f"{page.page}:{unit.start_line} ({unit.kind})"
        try:
            caught = _run_unit(unit, namespace)
        except BaseException as exc:  # noqa: BLE001 — report, don't mask
            if page.is_legacy:
                notes.append(
                    f"legacy page, failure not counted: {where}: {type(exc).__name__}"
                )
                continue
            raise AssertionError(
                f"documentation snippet failed: {where}\n{type(exc).__name__}: {exc}\n"
                "Fix the snippet, or mark an illustrative fragment with "
                "'<!-- skip-doctest: reason -->'."
            ) from exc
        for message in caught:
            if not _belongs_to_page(message, f"{page.page}:"):
                notes.append(
                    f"third-party warning in {where}: {message.category.__name__}"
                )
                continue
            if _is_allowed(message.category, unit.allowed_warnings):
                continue
            text = f"{where} emitted {message.category.__name__}: {message.message}"
            if page.is_legacy:
                notes.append(f"legacy page, warning not counted: {text}")
                continue
            raise AssertionError(
                f"{text}\nUse the current API, or list the category in "
                "'<!-- doctest-allow-warnings: Category -->' above the block "
                "if the page discusses the warning."
            )
        elapsed = time.perf_counter() - started
        if elapsed > 30.0:
            notes.append(f"slow unit {where}: {elapsed:.1f} s")
    for note in notes:
        print(f"[doc-snippets] {page.page}: {note}")


def test_markers_are_wellformed() -> None:
    """Skip markers carry a reason, module names are valid, checks are closed."""
    errors = [f"{p.page}: {e}" for p in ALL_PAGES for e in p.marker_errors]
    assert not errors, "\n".join(errors)


def test_pages_that_need_value_checks_have_them() -> None:
    """How-to pages assert on the numbers they show (10.88)."""
    missing = [
        p.page
        for p in EXECUTABLE_PAGES
        if p.page.startswith(REQUIRE_VALUE_CHECKS)
        and not any("assert" in u.source for u in p.executable)
    ]
    assert not missing, f"pages without a value check: {missing}"


def test_collection_is_healthy() -> None:
    """Guard against a silently vacuous harness (a parser regression that
    finds no blocks would make every page pass)."""
    n_units = sum(len(p.executable) for p in ALL_PAGES)
    assert len(EXECUTABLE_PAGES) >= 10, [p.page for p in EXECUTABLE_PAGES]
    assert n_units >= 40, n_units
    assert any(p.page == "README.md" for p in ALL_PAGES)


def test_legacy_prefixes_are_old_layout_paths_only() -> None:
    """The non-strict legacy handling must never cover a page of the new
    layout (getting-started, tutorials, how-to, reference, explanation,
    adr, audit)."""
    new_layout = (
        "docs/getting-started/",
        "docs/tutorials/",
        "docs/how-to/",
        "docs/reference/",
        "docs/explanation/",
    )
    for prefix in LEGACY_PREFIXES:
        assert not prefix.startswith(new_layout), prefix
        assert prefix.startswith("docs/"), prefix


def test_parser_handles_markers_and_checks(tmp_path) -> None:
    """The marker grammar on a synthetic page."""
    page = tmp_path / "docs" / "how-to" / "x.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# x\n\n"
        "<!-- doctest-module: helper_mod.py -->\n"
        "```python\nVALUE = 2\n```\n\n"
        "<!-- doctest-allow-warnings: UserWarning, DeprecationWarning -->\n\n"
        "```python\nimport helper_mod\nx = helper_mod.VALUE\n```\n\n"
        "<!-- skip-doctest: a sketch -->\n"
        "```python\nraise RuntimeError\n```\n\n"
        "<!-- doctest-check\nassert x == 2\n-->\n",
        encoding="utf-8",
    )
    original = globals()["REPO_ROOT"]
    try:
        globals()["REPO_ROOT"] = tmp_path
        parsed = parse_page(page)
    finally:
        globals()["REPO_ROOT"] = original
    kinds = [u.kind for u in parsed.units]
    assert kinds == ["module", "block", "check"]
    assert parsed.units[0].module_name == "helper_mod.py"
    assert parsed.units[1].allowed_warnings == ("UserWarning", "DeprecationWarning")
    assert parsed.skipped == [(16, "a sketch")]
    assert parsed.units[2].source == "assert x == 2"
    assert not parsed.marker_errors
