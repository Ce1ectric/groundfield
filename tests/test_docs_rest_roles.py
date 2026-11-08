"""The docs build renders the reST markup of the docstrings (finding 10.85).

``scripts/griffe_rest_roles.py`` is a Griffe extension that mkdocstrings
loads (``mkdocs.yml``): it turns Sphinx roles into optional autorefs and
``:math:`` / ``.. math::`` into arithmatex math, so the API reference no
longer shows ``:class:`World``` literally. These tests pin the conversion
rules and check it on the real package. Griffe ships with the docs
dependency group, so the tests skip without it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

griffe = pytest.importorskip("griffe")

from scripts.griffe_rest_roles import RestRoles, convert  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROLE = re.compile(r":(?:py:)?(?:class|func|meth|attr|mod|data|exc|obj|const|math):`")


def _identity(target: str) -> str:
    return target


@pytest.mark.finding("10.85")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            ":class:`~groundfield.World`",
            '<autoref identifier="groundfield.World" optional>'
            "<code>World</code></autoref>",
        ),
        (
            ":meth:`FieldResult.input_impedance`",
            '<autoref identifier="FieldResult.input_impedance" optional>'
            "<code>FieldResult.input_impedance()</code></autoref>",
        ),
        (
            ":py:func:`the fit <groundfield.vector_fit>`",
            '<autoref identifier="groundfield.vector_fit" optional>'
            "<code>the fit</code></autoref>",
        ),
        (":class:`!NotLinked`", "<code>NotLinked</code>"),
        (":math:`\\rho_1`", "$\\rho_1$"),
        (":ref:`some-label`", "some-label"),
    ],
)
def test_roles_become_autorefs_and_math(text: str, expected: str) -> None:
    assert convert(text, _identity) == expected


@pytest.mark.finding("10.85")
def test_role_across_a_line_break() -> None:
    text = (
        "see :meth:`Engine.solve\n    <groundfield.Engine.solve>`"
        " and :math:`Z(\\rho,\n f)`"
    )
    out = convert(text, _identity)
    assert 'identifier="groundfield.Engine.solve"' in out
    assert "$Z(\\rho, f)$" in out
    assert not ROLE.search(out)


@pytest.mark.finding("10.85")
def test_math_directive_becomes_a_display_block() -> None:
    text = (
        "Text.\n\n.. math::\n    :label: eq1\n\n    U_T = U_E - \\varphi\n\nMore text."
    )
    out = convert(text, _identity)
    assert out.split("\n")[2:5] == ["$$", "U_T = U_E - \\varphi", "$$"]
    assert ".. math::" not in out and ":label:" not in out
    assert out.endswith("More text.")


@pytest.mark.finding("10.85")
def test_code_and_doctests_are_left_alone() -> None:
    text = (
        "```python\nx = ':class:`A`'\n```\n\n>>> y = ':class:`B`'\n>>> y\n\n:class:`C`"
    )
    out = convert(text, _identity)
    assert ":class:`A`" in out and ":class:`B`" in out
    literal = "use ``:class:`X` `` in Sphinx, :class:`Y` here"
    assert convert(literal, _identity).startswith(
        "use ``:class:`X` `` in Sphinx, <autoref"
    )
    assert out.endswith('<autoref identifier="C" optional><code>C</code></autoref>')


@pytest.mark.finding("10.85")
def test_extension_converts_and_resolves_the_package_docstrings() -> None:
    """Every docstring of the package is free of roles after loading."""
    loader = griffe.GriffeLoader(
        extensions=griffe.load_extensions(RestRoles()),
        search_paths=[str(REPO / "src")],
    )
    package = loader.load("groundfield")
    loader.resolve_aliases(implicit=False, external=False)

    left = []
    stack = [package]
    while stack:
        obj = stack.pop()
        if obj.docstring is not None and ROLE.search(obj.docstring.value):
            left.append(obj.path)
        stack.extend(m for m in obj.members.values() if not m.is_alias)
    assert left == []

    # a short name is resolved in the scope of the documented object
    doc = package["solver.result.FieldResult.cluster_impedance"].docstring.value
    assert (
        'identifier="groundfield.solver.result.FieldResult.grounding_impedance"' in doc
    )
