"""The API reference covers the whole public API.

The public API of ``groundfield`` is the ``__all__`` list of the package
and of every public module (no dotted part of the module path starts
with an underscore). The reference pages under ``docs/reference/api/``
render it with mkdocstrings ``::: <path>`` directives. This test checks,
without building the site, that

- every name of every public ``__all__`` is rendered by a directive,
  directly or as a member of a rendered module;
- every such object has a docstring with a non-empty summary;
- every public function documents each of its parameters in a NumPy
  ``Parameters`` (or ``Other Parameters``) section and has a ``Returns``
  or ``Yields`` section when its annotation returns a value;
- every public method and property of a public class has a docstring.

mkdocstrings renders only objects that have a docstring
(``show_if_no_docstring`` is off), so a name without one is missing from
the reference even when a directive names it. The docstrings are read
from the source with :mod:`ast`, as griffe reads them: a constant or a
type alias is documented by a string literal right after its
assignment, not by a ``#:`` comment.

The rendering model of a directive follows mkdocstrings-python with the
options of ``mkdocs.yml``: a ``members`` list renders exactly those
members; without one, a module renders the public names of its
``__all__`` and the public objects defined in the module itself. A
submodule shadows an imported name of the same name (griffe sees
``groundfield.utils.suggest`` as the module, not as the function), and
an object rendered under one path covers every name bound to it
(``gf.World`` is ``groundfield.world.World``). The model was checked
against the ``objects.inv`` of a strict build: both find the same names.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import types
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

import groundfield

REPO = Path(__file__).resolve().parents[1]
API_DOCS = REPO / "docs" / "reference" / "api"

#: Public methods that stay undocumented on purpose: framework hooks
#: that are not called by users (pydantic's post-init hook).
UNDOCUMENTED_METHODS = frozenset({"model_post_init"})

_DIRECTIVE_RE = re.compile(r"^:::\s+(?P<path>[\w.]+)\s*$")
_SECTION_RE = re.compile(r"^(?P<title>[A-Z][A-Za-z ]+)\n-{3,}\s*$", re.MULTILINE)


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


@cache
def public_modules() -> tuple[str, ...]:
    """Dotted names of ``groundfield`` and of all its public modules."""
    names = ["groundfield"]
    for info in pkgutil.walk_packages(groundfield.__path__, "groundfield."):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.append(info.name)
    return tuple(sorted(names))


def _import(name: str) -> types.ModuleType:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return importlib.import_module(name)


@dataclass(frozen=True)
class PublicName:
    """One entry of a public ``__all__``."""

    module: str
    name: str

    @property
    def path(self) -> str:
        return f"{self.module}.{self.name}"

    @property
    def obj(self) -> object:
        return getattr(_import(self.module), self.name)


@cache
def public_names() -> tuple[PublicName, ...]:
    """Every public name of every public ``__all__`` (private names skipped)."""
    out = []
    for module in public_modules():
        for name in getattr(_import(module), "__all__", ()):
            if not _is_private(name):
                out.append(PublicName(module, name))
    return tuple(out)


# ---------------------------------------------------------------------
# Static docstrings (as griffe reads them)
# ---------------------------------------------------------------------


@cache
def _module_ast(module: str) -> ast.Module:
    path = _import(module).__file__
    assert path is not None, module
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _assigned_names(node: ast.stmt) -> list[str]:
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.TypeAlias)):
        target = node.target if isinstance(node, ast.AnnAssign) else node.name
        return [target.id] if isinstance(target, ast.Name) else []
    return []


def _resolve_from(module: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package = (
        module
        if _import(module).__file__.endswith("__init__.py")
        else module.rpartition(".")[0]
    )
    for _ in range(node.level - 1):
        package = package.rpartition(".")[0]
    return f"{package}.{node.module}" if node.module else package


def _walk_top(body: list[ast.stmt]) -> Iterator[tuple[int, list[ast.stmt]]]:
    """Statements of a module body, descending into ``if`` / ``try`` blocks."""
    for index, node in enumerate(body):
        yield index, body
        if isinstance(node, ast.If):
            yield from _walk_top(node.body)
            yield from _walk_top(node.orelse)
        elif isinstance(node, ast.Try):
            yield from _walk_top(node.body)


def data_docstring(module: str, name: str, _depth: int = 0) -> str | None:
    """Docstring of a module-level constant or type alias (string after it).

    Follows ``from x import name`` to the defining module, as griffe
    follows aliases.
    """
    tree = _module_ast(module)
    for index, body in _walk_top(tree.body):
        node = body[index]
        if name in _assigned_names(node):
            nxt = body[index + 1] if index + 1 < len(body) else None
            if (
                isinstance(nxt, ast.Expr)
                and isinstance(nxt.value, ast.Constant)
                and isinstance(nxt.value.value, str)
            ):
                return inspect.cleandoc(nxt.value.value)
            return None
    if _depth < 5:
        for index, body in _walk_top(tree.body):
            node = body[index]
            if isinstance(node, ast.ImportFrom) and any(
                (alias.asname or alias.name) == name for alias in node.names
            ):
                source = _resolve_from(module, node)
                original = next(
                    a.name for a in node.names if (a.asname or a.name) == name
                )
                return data_docstring(source, original, _depth + 1)
    return None


def _definition(obj: object) -> ast.AST | None:
    """AST node of the class or function definition of ``obj``."""
    obj = inspect.unwrap(obj) if callable(obj) else obj
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if module is None or qualname is None or module not in sys.modules:
        return None
    try:
        tree = _module_ast(module)
    except (AssertionError, OSError, TypeError):
        return None
    nodes: list[ast.AST] = [tree]
    for part in qualname.split("."):
        found = None
        for parent in nodes:
            for child in ast.iter_child_nodes(parent):
                if (
                    isinstance(
                        child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and child.name == part
                ):
                    found = child  # the last definition wins (overloads, redefinitions)
        if found is None:
            return None
        nodes = [found]
    return nodes[0]


def docstring_of(item: PublicName) -> str | None:
    """Static docstring of a public name, ``None`` if it has none."""
    obj = item.obj
    if isinstance(obj, types.ModuleType):
        return ast.get_docstring(_module_ast(obj.__name__))
    if inspect.isclass(obj) or inspect.isroutine(obj):
        node = _definition(obj)
        return ast.get_docstring(node) if node is not None else inspect.getdoc(obj)
    return data_docstring(item.module, item.name)


def summary(doc: str | None) -> str:
    """First paragraph of a docstring, stripped."""
    if not doc:
        return ""
    return doc.strip().split("\n\n", 1)[0].strip()


def numpy_sections(doc: str) -> dict[str, str]:
    """NumPy sections of a docstring: title -> body."""
    matches = list(_SECTION_RE.finditer(doc))
    out = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
        out[match["title"].strip()] = doc[match.end() : end]
    return out


def documented_parameters(doc: str) -> set[str]:
    """Names listed in the ``Parameters`` and ``Other Parameters`` sections."""
    names: set[str] = set()
    sections = numpy_sections(doc)
    for title in ("Parameters", "Other Parameters"):
        body = sections.get(title, "")
        for line in body.splitlines():
            if not line.strip() or line[:1].isspace():
                continue
            head = line.split(":", 1)[0]
            for part in head.split(","):
                names.add(part.strip().lstrip("*"))
    return names


# ---------------------------------------------------------------------
# What the reference pages render
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Directive:
    """One ``::: path`` directive of a reference page."""

    page: str
    path: str
    members: tuple[str, ...] | None


def _parse_members(block: list[str]) -> tuple[str, ...] | None:
    for i, line in enumerate(block):
        match = re.match(r"^(\s*)members:\s*(.*)$", line)
        if match is None:
            continue
        rest = match[2].strip()
        if rest.startswith("["):
            return tuple(
                m.strip().strip("'\"") for m in rest.strip("[]").split(",") if m.strip()
            )
        if rest in ("false", "False", "no"):
            return ()
        indent = len(match[1])
        members = []
        for item in block[i + 1 :]:
            item_match = re.match(r"^(\s*)-\s*(\S+)\s*$", item)
            if item_match is None or len(item_match[1]) < indent:
                break
            members.append(item_match[2].strip("'\""))
        return tuple(members)
    return None


@cache
def directives() -> tuple[Directive, ...]:
    """Every mkdocstrings directive of ``docs/reference/api/*.md``."""
    out = []
    for page in sorted(API_DOCS.glob("*.md")):
        lines = page.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            match = _DIRECTIVE_RE.match(line)
            if match is None:
                continue
            block = []
            for follow in lines[i + 1 :]:
                if follow.strip() and not follow[:1].isspace():
                    break
                block.append(follow)
            out.append(Directive(page.name, match["path"], _parse_members(block)))
    return tuple(out)


def _resolve(path: str) -> object:
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj: object = _import(".".join(parts[:cut]))
        except ImportError:
            continue
        for part in parts[cut:]:
            obj = getattr(obj, part)
        return obj
    raise ImportError(path)


def _defined_public_names(module: types.ModuleType) -> list[str]:
    """Public classes, functions and assignments defined in the module itself."""
    names = []
    for index, body in _walk_top(_module_ast(module.__name__).body):
        node = body[index]
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
        else:
            names.extend(_assigned_names(node))
    return [n for n in names if not _is_private(n) and n != "__all__"]


def _member(module: types.ModuleType, name: str) -> object:
    """The member ``name`` of ``module`` as griffe sees it.

    A submodule shadows an imported name of the same name:
    ``groundfield.utils.suggest`` is the module for griffe, although the
    package binds the attribute to the function ``suggest``.
    """
    if hasattr(module, "__path__"):
        submodule = f"{module.__name__}.{name}"
        if importlib.util.find_spec(submodule) is not None:
            return _import(submodule)
    return getattr(module, name, None)


@cache
def rendered() -> tuple[tuple[str, object, str], ...]:
    """``(name, object, page)`` of everything the directives render."""
    out = []
    for directive in directives():
        target = _resolve(directive.path)
        out.append((directive.path.rpartition(".")[2], target, directive.page))
        if isinstance(target, types.ModuleType):
            if directive.members is not None:
                names = list(directive.members)
            else:
                names = [
                    n for n in getattr(target, "__all__", ()) if not _is_private(n)
                ]
                names += _defined_public_names(target)
            for name in names:
                member = _member(target, name)
                if member is not None:
                    out.append((name, member, directive.page))
    return tuple(out)


_VALUE_TYPES = (int, float, complex, str, bytes, tuple, frozenset, type(None))


def _has_identity(obj: object) -> bool:
    """Whether ``is`` identifies the object (not an interned or shared value)."""
    return not isinstance(obj, _VALUE_TYPES)


def is_rendered(item: PublicName) -> bool:
    """Whether a directive renders the object of ``item``."""
    obj = item.obj
    for name, other, _ in rendered():
        if other is obj and (_has_identity(obj) or name == item.name):
            return True
    return False


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------


def test_public_api_is_enumerated() -> None:
    names = {item.path for item in public_names()}
    assert len(public_modules()) > 50
    for expected in (
        "groundfield.World",
        "groundfield.solver.solve_mom",
        "groundfield.utils.get_logger",
        "groundfield.generators.layout.LvNetworkLayout",
    ):
        assert expected in names


def test_directives_resolve() -> None:
    assert directives(), "no mkdocstrings directive found"
    for directive in directives():
        target = _resolve(directive.path)
        for member in directive.members or ():
            assert hasattr(target, member), (
                f"{directive.page}: {directive.path}.{member}"
            )


def test_every_public_name_is_rendered() -> None:
    missing = sorted(item.path for item in public_names() if not is_rendered(item))
    assert missing == [], (
        f"{len(missing)} public names are on no reference page "
        f"(add a ::: directive or a members entry in docs/reference/api/): {missing}"
    )


def test_every_public_name_has_a_docstring_summary() -> None:
    missing = sorted(
        {item.path for item in public_names() if not summary(docstring_of(item))}
    )
    assert missing == [], (
        f"{len(missing)} public names have no docstring summary (mkdocstrings "
        f"does not render them; a constant needs a string literal after it): {missing}"
    )


def _public_functions() -> dict[str, object]:
    out: dict[str, object] = {}
    for item in public_names():
        obj = item.obj
        if inspect.isfunction(obj):
            out.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
    return out


def test_public_functions_document_parameters_and_returns() -> None:
    problems = []
    for path, func in sorted(_public_functions().items()):
        node = _definition(func)
        doc = ast.get_docstring(node) if node is not None else None
        if not doc:
            continue  # reported by the summary test
        params = [
            p.name
            for p in inspect.signature(func).parameters.values()
            if p.name not in ("self", "cls")
        ]
        missing = [p for p in params if p not in documented_parameters(doc)]
        if missing:
            problems.append(f"{path}: parameters {missing}")
        returns = (
            node.returns
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )
        if returns is not None and ast.unparse(returns) not in ("None", "NoReturn"):
            sections = numpy_sections(doc)
            if "Returns" not in sections and "Yields" not in sections:
                problems.append(
                    f"{path}: no Returns section (-> {ast.unparse(returns)})"
                )
    assert problems == [], "\n".join(problems)


def test_public_methods_have_docstrings() -> None:
    missing = []
    classes = {}
    for item in public_names():
        obj = item.obj
        if inspect.isclass(obj) and obj.__module__.startswith("groundfield"):
            classes.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)
    for path, cls in sorted(classes.items()):
        node = _definition(cls)
        if node is None:
            continue
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_private(child.name) or child.name.startswith("__"):
                continue
            if child.name in UNDOCUMENTED_METHODS:
                continue
            if any(
                isinstance(d, ast.Attribute) and d.attr in ("setter", "deleter")
                for d in child.decorator_list
            ):
                continue
            if not ast.get_docstring(child):
                missing.append(f"{path}.{child.name}")
    assert sorted(set(missing)) == [], (
        f"public methods without docstring: {sorted(set(missing))}"
    )


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (
            "Summary.\n\nParameters\n----------\n"
            "a, b : float\n    Text.\n*args\n    More.\n",
            {"a", "b", "args"},
        ),
        ("Summary.\n\nReturns\n-------\nfloat\n", set()),
    ],
)
def test_parameter_parser(doc: str, expected: set[str]) -> None:
    assert documented_parameters(doc) == expected
