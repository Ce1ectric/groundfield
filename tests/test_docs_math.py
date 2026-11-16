"""Math in the documentation renders: guards against the known breakages.

The site renders math with MathJax 3.2.2 (vendored) from the arithmatex
output of the Markdown pages, the notebooks and the docstrings. A browser
check of the built site before the release of 2026.11.0 (headless Chromium
under the mike version path, all 99 pages: 4545 formulas typeset, none with
a MathJax error) found three classes of breakage that the strict MkDocs build
does not report. This module keeps them out of the sources:

* **math in a heading** shows as raw TeX in the table of contents, the
  navigation and the search index;
* **an inline formula broken over two lines whose second line starts with
  ``+``, ``-``, ``*`` or ``1.``**: Markdown reads that line as a list item
  and splits the formula, which then stays raw text (two docstrings of
  ``groundfield.coupling`` did this);
* **an unescaped underscore inside** ``\\texttt{…}``, which MathJax refuses in
  text mode (``'_' allowed only in math mode``).
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SRC = REPO / "src" / "groundfield"

#: The audit history quotes old texts verbatim; it is not checked.
SKIP_DIRS = ("audit",)

LIST_START = re.compile(r"^\s*(?:[-+*]\s|\d+\.\s)")
TEXTTT_UNDERSCORE = re.compile(r"\\texttt\{[^}]*(?<!\\)_[^}]*\}")


def _markdown_pages() -> Iterator[tuple[str, str]]:
    for path in sorted(DOCS.rglob("*.md")):
        rel = path.relative_to(REPO).as_posix()
        if any(f"docs/{d}/" in rel for d in SKIP_DIRS):
            continue
        yield rel, path.read_text(encoding="utf-8")


def _notebook_markdown() -> Iterator[tuple[str, str]]:
    for path in sorted(DOCS.rglob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for k, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                src = cell["source"]
                text = src if isinstance(src, str) else "".join(src)
                yield f"{path.relative_to(REPO).as_posix()}:cell {k}", text


def _public(name: str) -> bool:
    return not name.startswith("_") or name.startswith("__")


def _docstrings() -> Iterator[tuple[str, str]]:
    """Docstrings the API reference renders: public modules, classes, functions."""
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC).with_suffix("").parts
        if not all(_public(p) for p in parts):
            continue
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree, clean=True)
        if doc:
            yield f"{rel}:<module>", doc
        stack = [(tree, "")]
        while stack:
            parent, prefix = stack.pop()
            for node in parent.body:
                if not isinstance(
                    node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    continue
                if not _public(node.name):
                    continue
                doc = ast.get_docstring(node, clean=True)
                if doc:
                    yield f"{rel}:{prefix}{node.name}", doc
                if isinstance(node, ast.ClassDef):
                    stack.append((node, f"{node.name}."))


def _prose_lines(text: str) -> Iterator[str]:
    """Lines outside fenced code blocks and outside ``$$`` display blocks."""
    fence = None
    display = False
    for line in text.splitlines():
        stripped = line.strip()
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if stripped.startswith("$$") or stripped.endswith("$$"):
            if stripped.count("$$") == 1:
                display = not display
            continue
        if display:
            continue
        yield line


def _inline_dollars(line: str) -> int:
    """Number of single ``$`` delimiters on a line (no ``$$``, no ``\\$``)."""
    line = re.sub(r"``.*?``", "", line)  # reST inline literals
    line = re.sub(r"`[^`]*`", "", line)  # Markdown inline code
    line = line.replace("$$", "")
    return len(re.findall(r"(?<!\\)\$", line))


def _split_inline_math(text: str) -> list[str]:
    problems = []
    open_math = False
    previous = ""
    for line in _prose_lines(text):
        if open_math and LIST_START.match(line):
            problems.append(f"{previous.strip()[:60]} / {line.strip()[:60]}")
        if _inline_dollars(line) % 2:
            open_math = not open_math
        if not line.strip():
            open_math = False
        previous = line
    return problems


SOURCES = [*_markdown_pages(), *_notebook_markdown()]


@pytest.mark.parametrize("where,text", SOURCES, ids=[w for w, _ in SOURCES])
def test_no_math_in_headings(where: str, text: str) -> None:
    bad = [
        line
        for line in _prose_lines(text)
        if re.match(r"^#{1,6}\s", line) and ("$" in line or "\\(" in line)
    ]
    assert not bad, f"{where}: math in a heading: {bad}"


def test_no_inline_formula_is_split_by_a_list_marker() -> None:
    found = []
    for where, text in [*SOURCES, *_docstrings()]:
        found += [f"{where}: {p}" for p in _split_inline_math(text)]
    assert not found, "\n".join(found)


def test_no_underscore_in_texttt() -> None:
    found = []
    for where, text in [*SOURCES, *_docstrings()]:
        found += [f"{where}: {m.group(0)}" for m in TEXTTT_UNDERSCORE.finditer(text)]
    assert not found, "\n".join(found)
