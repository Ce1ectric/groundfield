r"""Griffe extension: render the reST markup of the docstrings in the docs site.

The docstrings of ``groundfield`` follow the NumPy convention and use
Sphinx markup: roles such as :class:`~groundfield.World` or
:math:`\rho`, and ``.. math::`` blocks. MkDocs renders docstrings as
Markdown, where that markup shows up literally (review pass 10, finding
10.85: about 1 400 literal roles in the API reference). This extension
rewrites the markup while mkdocstrings loads the package; the source
docstrings stay unchanged and keep working in ``help()``, IPython and
Sphinx.

Conversions
-----------
- ``:class:``, ``:func:``, ``:meth:``, ``:attr:``, ``:mod:``,
  ``:data:``, ``:exc:``, ``:obj:``, ``:const:`` (also with the ``py:``
  domain, the ``~`` and ``!`` prefixes and the ``title <target>`` form)
  become *optional* mkdocs-autorefs references: a link when the target
  is rendered somewhere in the site, inline code otherwise. A short
  name is resolved in the scope of the documented object (its members,
  imports and parents), then as a top-level name of ``groundfield``.
- ``:math:`x``` becomes ``$x$`` (pymdownx.arithmatex).
- A ``.. math::`` block becomes a ``$$`` display block.
- ``:ref:`` and ``:term:`` become their plain text.

Fenced code blocks and doctest blocks (``>>>`` up to the next blank
line) are left unchanged.

Usage
-----
In ``mkdocs.yml``::

    plugins:
      - mkdocstrings:
          handlers:
            python:
              options:
                extensions:
                  - scripts/griffe_rest_roles.py

The path is resolved relative to ``mkdocs.yml``.
"""

from __future__ import annotations

import html
import re
import textwrap
from collections.abc import Callable, Iterator
from typing import Any

import griffe

__all__ = ["RestRoles", "convert"]

#: Roles that name a Python object.
PY_ROLES = frozenset(
    {"class", "func", "meth", "attr", "mod", "data", "exc", "obj", "const"}
)

_ROLE_RE = re.compile(
    r":(?:py:)?(?P<role>class|func|meth|attr|mod|data|exc|obj|const|math|ref|term):"
    r"`(?P<body>[^`]+)`"
)
_TITLE_RE = re.compile(r"(?P<title>.+?)\s*<(?P<target>[^<>]+)>\s*")
_MATH_DIRECTIVE_RE = re.compile(r"^(?P<indent>[ \t]*)\.\. math::[ \t]*(?P<inline>.*)$")
_DIRECTIVE_OPTION_RE = re.compile(r"^\s*:[a-z-]+:")

Resolver = Callable[[str], "str | None"]


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _role(match: re.Match[str], resolve: Resolver | None) -> str:
    """Return the replacement text of one role."""
    role = match["role"]
    # A role may continue on the next docstring line; Markdown does not
    # care about the line break, the identifier must not contain it.
    body = " ".join(match["body"].split())
    if role == "math":
        return f"${body}$"
    title = None
    titled = _TITLE_RE.fullmatch(body)
    if titled is not None:
        title, body = titled["title"].strip(), titled["target"].strip()
    if role not in PY_ROLES:  # :ref:, :term:
        return html.escape(title or body)
    suppress = body.startswith("!")
    short = body.lstrip("!").startswith("~")
    written = body.lstrip("!~").strip()
    target = written.split("(", 1)[0].strip()
    if title:
        display = title
    elif "(" in written:
        display = written.rsplit(".", 1)[-1] if short else written
    else:
        display = target.rsplit(".", 1)[-1] if short else target
        if role in ("func", "meth"):
            display += "()"
    code = f"<code>{html.escape(display)}</code>"
    if suppress or not target:
        return code
    identifier = (resolve(target) if resolve is not None else None) or target
    attribute = html.escape(identifier, quote=True)
    return f'<autoref identifier="{attribute}" optional>{code}</autoref>'


def convert(text: str, resolve: Resolver | None = None) -> str:
    """Rewrite the reST roles and ``.. math::`` blocks of a docstring.

    Parameters
    ----------
    text
        Docstring text (already de-indented by the loader).
    resolve
        Maps a role target to the identifier of the documented object,
        or returns ``None`` when it cannot. Without it the target is used
        as written.

    Returns
    -------
    str
        The docstring with Markdown math and autorefs instead of reST.
    """
    if ":" not in text and ".. math::" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    pending: list[str] = []  # unprotected lines, converted together

    def flush() -> None:
        if pending:
            joined = "\n".join(pending)
            out.extend(_ROLE_RE.sub(lambda m: _role(m, resolve), joined).split("\n"))
            pending.clear()

    in_fence = False
    in_doctest = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            flush()
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        if stripped.startswith(">>>"):
            in_doctest = True
        elif not stripped:
            in_doctest = False
        if in_doctest:
            flush()
            out.append(line)
            i += 1
            continue
        directive = _MATH_DIRECTIVE_RE.match(line)
        if directive is not None:
            flush()
            indent = directive["indent"]
            block: list[str] = []
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or _indent_of(lines[j]) > len(indent)
            ):
                block.append(lines[j])
                j += 1
            while block and not block[-1].strip():
                block.pop()
            body = [
                b
                for b in textwrap.dedent("\n".join(block)).split("\n")
                if not _DIRECTIVE_OPTION_RE.match(b)
            ]
            inline = directive["inline"].strip()
            content = ([inline] if inline else []) + body
            while content and not content[0].strip():
                content.pop(0)
            if out and out[-1].strip():
                out.append("")
            out.append(f"{indent}$$")
            out.extend(f"{indent}{c}" if c.strip() else "" for c in content)
            out.append(f"{indent}$$")
            out.append("")
            i = j
            continue
        pending.append(line)
        i += 1
    flush()
    return "\n".join(out)


def _resolver(obj: griffe.Object, collection: Any, package: str) -> Resolver:
    """Resolve role targets in the scope of ``obj``."""

    def resolve(target: str) -> str | None:
        candidates: list[str] = []
        if target.startswith(f"{package}."):
            candidates.append(target)
        else:
            head, _, rest = target.partition(".")
            try:
                full = obj.resolve(head)
            except Exception:  # noqa: BLE001 - unresolvable names stay code
                full = None
            if full:
                candidates.append(f"{full}.{rest}" if rest else full)
            candidates.append(f"{package}.{target}")
        for candidate in candidates:
            try:
                found = collection[candidate]
            except Exception:  # noqa: BLE001
                continue
            try:
                return str(found.canonical_path)
            except Exception:  # noqa: BLE001 - unresolvable alias
                return candidate
        return None

    return resolve


def _objects(obj: griffe.Object) -> Iterator[griffe.Object]:
    """Yield the object and every member defined in it (aliases excluded)."""
    yield obj
    for member in obj.members.values():
        if member.is_alias:
            continue
        yield from _objects(member)  # type: ignore[arg-type]


class RestRoles(griffe.Extension):
    """Convert the reST markup of every docstring of a loaded package."""

    def on_package(
        self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any
    ) -> None:
        """Rewrite the docstrings once the package is completely loaded."""
        collection = loader.modules_collection
        for obj in _objects(pkg):
            docstring = obj.docstring
            if docstring is None or not docstring.value:
                continue
            docstring.value = convert(
                docstring.value, _resolver(obj, collection, pkg.name)
            )
