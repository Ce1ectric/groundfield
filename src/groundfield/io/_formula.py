r"""Safe parsing, evaluation and rendering of impedance-formula strings.

A ``groundinsight`` ``BusType`` carries its impedance as a text formula
$Z(f, \rho)$ in the frequency ``f`` (Hz) and the soil resistivity ``rho``
($\Omega\,\mathrm{m}$), with ``j`` (or ``I``) as the imaginary unit, e.g.

```text
0.0162429242886*rho - f*(3.39608430625e-5 + 0.00011948072096*j)
```

Such text arrives from JSON files and databases and is therefore
*untrusted input*. SymPy's string parser (``sympify`` / ``parse_expr``)
evaluates the text as Python code, so a crafted formula can run
arbitrary code or exhaust memory (review finding 10.01). This module
never hands text to ``sympify``, ``parse_expr``, ``eval`` or ``exec``:
:func:`parse_impedance_formula` parses the text with :func:`ast.parse`,
which executes nothing, checks the syntax tree against an allow-list and
**builds** the SymPy expression node by node without letting SymPy
evaluate anything; :func:`formula_function` turns the expression into a
NumPy function without SymPy's code generation.
:func:`format_impedance_formula` is the inverse direction: it renders a
SymPy expression in the same grammar with a fixed number of significant
digits per coefficient. :func:`round_coefficients` applies the same
rounding to the expression itself; it is the rounding step of the one
formula renderer that ``fit_to_sympy``, ``fit_to_sympy_standard`` and
the ``groundinsight`` exporter share (review pass 10, 10.73).

Grammar
-------
After Unicode NFKC normalisation the text may contain only ASCII
letters, digits, ``_ . + - * / ( )``, spaces and tabs, and must be one
Python expression built from

* numeric literals: integers, finite floats and imaginary literals such
  as ``2.5j``;
* names: the caller's free symbols (default ``f`` and ``rho``), the
  imaginary unit ``j`` or ``I``, and the constants ``pi`` and ``E``;
* operators: unary ``+`` / ``-`` and binary ``+ - * /``;
* powers ``base ** n`` whose exponent ``n`` is a numeric literal
  (``2``, ``-1``, ``0.5``, ``(1/2)``, ``(-3/2)``, the sign before the
  fraction or its numerator) with $|n| \le$ :data:`MAX_POW`; the
  exponents of nested powers multiply, through sums, products and
  function calls, and their product is bounded by :data:`MAX_POW` as
  well (``(f**8)**8`` is fine, ``((f**8)**8)**2`` and
  ``((f + j)**64 + 1)**2`` are not; ``sqrt`` counts as ``1/2``);
* calls ``sqrt(x)``, ``exp(x)`` and ``log(x)`` with exactly one
  positional argument; ``log`` may not appear inside the argument of
  ``exp`` (``exp(c*log(x))`` is the power ``x**c`` with an exponent ``c``
  that is no literal; groundinsight refuses it as well).

How the expression is built
---------------------------
SymPy simplifies while it constructs an expression: ``Pow(b, e)`` asks
for the real part and the argument of ``b``, ``exp(x)`` asks whether
the factors of ``x`` are comparable and expands powers of complex sums
to find out, an integer under a fractional power is factorised, ``log``
of ``r + r*j`` takes ``sqrt(2*r**2)``. None of these algorithms has a
cost bound in the size of the formula; three rounds of rules against
particular inputs left others open (the second audit of the calendar
release found 67- and 75-character formulas that ran for more than a
minute). So the parser never lets SymPy evaluate:

* every node is created with ``evaluate=False`` (``Add``, ``Mul``,
  ``Pow``, ``exp``, ``log``), so SymPy only stores its arguments; the
  parser asks SymPy no questions about the expression (no assumptions,
  no ``subs``, ``simplify``, ``expand`` or ``evalf``);
* numbers are computed by the parser itself, in floating point
  (:mod:`mpmath`, 128 bits), with the bounds below checked at every
  step: every symbol-free sub-expression becomes one number, stored as
  SymPy floats of 17 significant digits (real and imaginary part,
  SymPy's principal branch for roots and logarithms; parts below
  ``2**-100`` times the magnitude are dropped), so ``3/4``, ``sqrt(2)``,
  ``log(2)`` and ``E**2`` become floats;
* within one product ``a*b/c*...`` (also across parentheses and unary
  minus) the numeric factors are multiplied into one coefficient, so
  ``2*pi*f*j/1000`` becomes ``0.0062831853071795865*I*f``; within one
  sum ``a + b - c + ...`` the numbers are added and terms with equal
  factors (in the same order) are collected, so ``f - f + 1000`` is
  the number 1000; nothing else is rearranged: no expansion, no
  distribution over sums, no merging of powers;
* the result contains only ``Add``, ``Mul``, ``Pow`` (literal exponent),
  ``exp``, ``log``, SymPy floats, the integer ``-1``, ``I`` and the
  caller's symbols.

Every node therefore costs a bounded amount of work, and the time and
memory of a parse, accepted or rejected, grow linearly with the size of
the formula. Every number of the expression is the float64 nearest to
its value computed at 128 bits (relative error at most ``1.1e-16``).

Do not hand an expression parsed from untrusted text to SymPy functions
that evaluate it: ``simplify``, ``expand``, ``subs``, ``doit``, ``evalf``
of sub-expressions, but also ``lambdify`` and SymPy's printers (``str``,
``repr``, LaTeX), which order and negate sub-expressions with SymPy's
evaluation switched on and so run the algorithms the parser avoids
(for a 1 439-character formula within every limit ``str()`` took
4.9 s and ``lambdify`` 6.8 s, :func:`formula_function` 2 ms).
:func:`formula_function` evaluates a parsed expression with NumPy; to
show a formula, show its text.

Resource bounds
---------------
* at most :data:`MAX_CHARS` characters, :data:`MAX_NODES` syntax-tree
  nodes and a nesting depth of :data:`MAX_DEPTH`;
* every numeric literal has at most :data:`MAX_LITERAL_CHARS`
  characters, its exact decimal value is checked (``1e-<320 nines>``
  is refused, not read as zero);
* every number the parser computes or keeps (literals, coefficients,
  values of symbol-free sub-expressions) is zero or lies within
  $10^{-300} \le |x| \le 10^{300}$ (:data:`MAX_LOG10_MAGNITUDE`); a
  power ``base**n`` must not create a larger number from the numbers in
  its base (``n`` times their largest ``|log10|`` at most 300: the
  bound SymPy's distribution of ``(c*f)**n`` into ``c**n * f**n`` would
  reach); ``exp`` of a number takes arguments up to about 690.8 in
  magnitude;
* the bookkeeping of one parse is capped at :data:`MAX_WORK` steps;
* division by zero, ``0 ** -n`` and ``log(0)`` are refused, also when the
  zero comes from collected terms (``1/(f - f)``).

Exact numbers
-------------
groundinsight checks the same formulas with SymPy's own (exact)
arithmetic; two of its limits on exact numbers are kept so that both
parsers refuse the same strings of the shared formula corpus. A number is
*exact* if it is written with integer literals (``3``, ``2j``) and
``+ - * /`` and integer powers of exact numbers; ``pi``, ``E`` and ``j``
may appear as factors of an exact coefficient (``2*pi`` has the exact
coefficient 2), float literals, roots and function values are not exact.

* A fractional power ``p/q`` (also ``sqrt``, ``q = 2``) of an exact
  number, i.e. of an exact base or of the exact coefficient ``c`` of a
  product base ``c*f*...``, needs $q \le$ :data:`MAX_ROOT_DENOMINATOR`
  and (digits of the number) $\times\, q \le$ :data:`MAX_RADICAND_DIGITS`
  (``12**(1/3)`` and ``(2*f)**(3/2)`` are fine, a 300-digit integer under
  ``**(63/64)`` is not; written as a float, ``2.0**(1/100)``, a number is
  not limited). SymPy would factorise such numbers.
* The numerator and denominator of an exact fraction have at most
  :data:`MAX_RATIONAL_DIGITS` digits, and so has the least common
  multiple of the denominators of the exact coefficients of one sum
  (SymPy would add them exactly). The formulas written by every
  ``groundfield`` exporter up to 0.15.0 (rational coefficients with up
  to 254-digit integers) and by the current one stay inside these limits.
"""

from __future__ import annotations

import ast
import math
import numbers
import re
import string
import unicodedata
from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import Any, NamedTuple

import mpmath
import numpy as np
import sympy as sp
from mpmath.libmp import from_rational, round_nearest
from sympy.printing.str import StrPrinter

__all__ = [
    "FormulaError",
    "parse_impedance_formula",
    "formula_function",
    "format_impedance_formula",
    "round_coefficients",
    "MAX_CHARS",
    "MAX_NODES",
    "MAX_DEPTH",
    "MAX_LITERAL_CHARS",
    "MAX_POW",
    "MAX_ROOT_DENOMINATOR",
    "MAX_RADICAND_DIGITS",
    "MAX_LOG10_MAGNITUDE",
    "MAX_RATIONAL_DIGITS",
    "MAX_WORK",
]

MAX_CHARS = 10_000
"""Maximum length of a formula in characters (before and after NFKC)."""

MAX_NODES = 2_000
"""Maximum number of nodes of the Python syntax tree.

The longest formula written by the 0.15.0 exporter has 239 nodes.
"""

MAX_DEPTH = 100
"""Maximum nesting depth of the Python syntax tree."""

MAX_LITERAL_CHARS = 400
"""Maximum length of a single numeric literal in characters."""

MAX_POW = 64
"""Maximum magnitude of a literal exponent in ``base ** n``.

Also the bound on the product of the exponents of nested powers
(``(f**8)**8`` has the product 64), the limit groundinsight's guard
applies as well.
"""

MAX_ROOT_DENOMINATOR = 64
"""Maximum denominator ``q`` of a fractional exponent ``p/q`` of an exact number.

Applies to an exact base and to the exact coefficient of a product base
(``sqrt`` has ``q = 2``); the same limit as groundinsight's formula guard.
"""

MAX_RADICAND_DIGITS = 300
"""Maximum digits times ``q`` of an exact number under a fractional power ``p/q``.

Digits of a fraction count numerator and denominator. SymPy would keep a
radicand of up to ``q - 1`` times the digits of the number and test it for
primality: ``27*p*q`` under ``63/64`` takes 3 ms with two 3-digit primes
``p``, ``q`` (7 digits), 0.33 s with two 10-digit primes, 2.6 s with two
20-digit primes and more than 15 s with two 50-digit primes. The parser
never lets SymPy do this (see the module notes); the limit keeps absurd
exact numbers out and is the same as in groundinsight's formula guard.
"""

MAX_LOG10_MAGNITUDE = 300.0
"""Bound on ``|log10|x||`` for every non-zero number the parser computes or keeps."""

MAX_RATIONAL_DIGITS = 1_000
"""Maximum digits of the numerator and denominator of an exact fraction.

Formulas written by the 0.15.0 exporter contain fractions with up to
254-digit numerators and denominators.
"""

MAX_WORK = 200_000
"""Maximum bookkeeping steps of one parse.

A step is one node that is built, one factor or term that is carried
into a product or a sum, or one argument of a SymPy node. The longest
exporter formulas need a few thousand.
"""

_SHADOW_PREC = 128
"""Binary precision of the numbers the parser computes."""

_ZERO_SUM_BITS = 100
"""A sum below ``2**-100`` times its largest term counts as zero (not exact sums)."""

_MAX_EXP_ARGUMENT = MAX_LOG10_MAGNITUDE * math.log(10.0)  # about 690.8
_MAX_SIGNIFICANT_DIGITS = 17  # float64 carries at most 17 significant digits
_NUMERIC_DIGITS = 17
"""Significant digits of a float the parser stores (prints every float64 exactly)."""

_CONSTANT_NAMES = ("j", "I", "pi", "E")
_FUNCTIONS = ("sqrt", "exp", "log")
_ALLOWED_CHARACTERS = frozenset(string.ascii_letters + string.digits + "_.+-*/() \t")
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Call,
)
_CHARACTER_HINTS = {
    "^": "use '**' for powers",
    "−": "use the ASCII '-' for minus",
    "·": "use '*' for multiplication",
    "×": "use '*' for multiplication",
    "ρ": "write 'rho'",
    "ω": "write '2*pi*f'",
    "π": "write 'pi'",
    ",": "functions take exactly one argument",
}
_DECIMAL_LITERAL = re.compile(
    r"(?P<int>[0-9]*)(?:\.(?P<frac>[0-9]*))?(?:[eE](?P<exp>[+-]?[0-9]+))?"
)


class FormulaError(ValueError):
    """An impedance formula is outside the allowed grammar or resource bounds.

    Subclass of :class:`ValueError`. The message names the offending
    element and, where possible, its column in the normalised formula.
    """


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


def parse_impedance_formula(
    text: str,
    *,
    symbols: Iterable[str] = ("f", "rho"),
) -> sp.Expr:
    r"""Parse an impedance formula without executing or evaluating it.

    Parameters
    ----------
    text : str
        Formula, e.g. ``"0.04*rho + f*(1.5e-4 + 2.5e-7*j*rho)"``. ``f``
        is the frequency in Hz and ``rho`` the soil resistivity in
        $\Omega\,\mathrm{m}$ by the ``groundinsight`` convention; the
        parser itself attaches no units.
    symbols : iterable of str, optional
        Names of the free symbols that may appear (default ``f`` and
        ``rho``). They must be valid identifiers and must not shadow
        ``j``, ``I``, ``pi``, ``E``, ``sqrt``, ``exp`` or ``log``.

    Returns
    -------
    sympy.Expr
        Unevaluated expression that keeps the structure of the formula:
        symbol-free sub-expressions are SymPy floats, the numeric factors
        of a product form one coefficient, like terms of a sum are
        collected (see the module notes). Its free symbols are plain
        :class:`sympy.Symbol` objects named after a subset of
        ``symbols``; ``j`` and ``I`` become :data:`sympy.I`. Evaluate it
        with :func:`formula_function`, not with SymPy.

    Raises
    ------
    FormulaError
        If the text is not a string, violates the grammar or exceeds a
        resource bound (see the module docstring). No other exception
        escapes for any text.
    ValueError
        If ``symbols`` contains an invalid or reserved name.

    Notes
    -----
    Parsing never calls ``sympify``, ``parse_expr``, ``eval``, ``exec``
    or ``compile`` on the text (only :func:`ast.parse`), never lets SymPy
    evaluate a node (every node is built with ``evaluate=False``) and
    computes every number itself in bounded floating-point arithmetic. So
    the time and memory a formula costs, accepted or rejected, grow
    linearly with its size: audit 1 of the calendar release found sums and
    products that took 13 s to 105 s, the groundinsight stream rational
    powers of large integers that ran for more than 20 s, and audit 2
    67- and 75-character formulas (``exp`` of symbol-free powers of
    complex sums, fractional powers of symbolic powers) that ran for more
    than a minute, all while SymPy evaluated the expression being built.
    """
    names = _check_symbol_names(symbols)
    src = _normalise(text)
    tree = _parse_tree(src)
    _check_tree(tree, src, names)
    builder = _Builder(src, {name: sp.Symbol(name) for name in names})
    try:
        return builder.expression(builder.build(tree.body))
    except FormulaError:
        raise
    except RecursionError:  # pragma: no cover - guarded by MAX_DEPTH
        raise FormulaError("formula is nested too deeply") from None
    except (ArithmeticError, ValueError, TypeError, MemoryError) as exc:
        # Defensive: every bound is checked before it could be exceeded,
        # but no arithmetic error may escape as anything but FormulaError.
        raise FormulaError(
            f"formula cannot be evaluated within the bounds ({type(exc).__name__}: "
            f"{str(exc)[:200]})"
        ) from None


def _check_symbol_names(symbols: Iterable[str]) -> tuple[str, ...]:
    """Validate the caller's symbol names (a programming error if wrong)."""
    if isinstance(symbols, str):
        symbols = (symbols,)
    names = tuple(symbols)
    for name in names:
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
            raise ValueError(f"invalid symbol name {name!r}")
    reserved = sorted(set(names) & (set(_CONSTANT_NAMES) | set(_FUNCTIONS)))
    if reserved:
        raise ValueError(f"symbol names {reserved} are reserved")
    return names


def _describe_character(char: str) -> str:
    """Return ``'x' (U+0078)`` plus an optional hint."""
    hint = _CHARACTER_HINTS.get(char)
    desc = f"{char!r} (U+{ord(char):04X})"
    return f"{desc}: {hint}" if hint else desc


def _normalise(text: object) -> str:
    """NFKC-normalise, bound the length and restrict the character set."""
    if not isinstance(text, str):
        raise FormulaError(f"formula must be a string, got {type(text).__name__}")
    if len(text) > MAX_CHARS:
        raise FormulaError(f"formula has {len(text)} characters (limit {MAX_CHARS})")
    src = unicodedata.normalize("NFKC", text)
    if len(src) > MAX_CHARS:
        raise FormulaError(
            f"formula has {len(src)} characters after NFKC normalisation "
            f"(limit {MAX_CHARS})"
        )
    bad = sorted({c for c in src if c not in _ALLOWED_CHARACTERS})
    if bad:
        shown = "; ".join(_describe_character(c) for c in bad[:5])
        raise FormulaError(
            f"character(s) not allowed in an impedance formula: {shown}. "
            "Allowed are ASCII letters, digits, '_', '.', '+', '-', '*', '/', "
            "'(', ')' and spaces"
        )
    src = src.strip()
    if not src:
        raise FormulaError("formula is empty")
    return src


def _parse_tree(src: str) -> ast.Expression:
    """Parse ``src`` as a single expression (executes nothing)."""
    try:
        return ast.parse(src, mode="eval")
    except SyntaxError as exc:
        where = f" at column {exc.offset}" if exc.offset else ""
        raise FormulaError(f"not a valid expression{where}: {exc.msg}") from None
    except (ValueError, MemoryError, RecursionError, OverflowError) as exc:
        raise FormulaError(f"not a valid expression: {exc}") from None


def _check_tree(tree: ast.Expression, src: str, names: tuple[str, ...]) -> None:
    """Check size, depth, node types, names and literal lengths before building.

    Also rejects ``log`` inside the argument of ``exp`` (see the module
    notes).
    """
    allowed_names = set(names) | set(_CONSTANT_NAMES) | set(_FUNCTIONS)
    unknown: set[str] = set()
    n_nodes = 0
    stack: list[tuple[ast.AST, int, bool]] = [(tree, 1, False)]
    while stack:
        node, depth, inside_exp = stack.pop()
        n_nodes += 1
        if n_nodes > MAX_NODES:
            raise FormulaError(f"formula has more than {MAX_NODES} syntax nodes")
        if depth > MAX_DEPTH:
            raise FormulaError(f"formula is nested deeper than {MAX_DEPTH} levels")
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError(
                f"{_node_label(node)} is not allowed{_column(node)}: "
                f"{_snippet(src, node)}"
            )
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            unknown.add(node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
            spelling = _literal_text(src, node)
            if len(spelling) > MAX_LITERAL_CHARS:
                raise FormulaError(
                    f"numeric literal longer than {MAX_LITERAL_CHARS} characters"
                    f"{_column(node)}: {_snippet(src, node)}"
                )
        called = _called_name(node)
        if called == "log" and inside_exp:
            raise FormulaError(
                f"log() inside exp() is not allowed{_column(node)}: "
                f"{_snippet(src, node)}; exp(c*log(x)) is the power x**c, "
                "whose exponent c is no literal. Write the power x**c with a "
                "literal exponent instead"
            )
        inside_exp = inside_exp or called == "exp"
        family = _chain_family(node)
        for child in ast.iter_child_nodes(node):
            same_chain = family is not None and child is getattr(node, "left", None)
            same_chain = same_chain and _chain_family(child) == family
            stack.append((child, depth if same_chain else depth + 1, inside_exp))
    if unknown:
        raise FormulaError(
            f"formula contains unrecognised free symbols {sorted(unknown)!r}; "
            f"allowed symbols are {', '.join(names) or '(none)'}, the imaginary "
            "unit j (or I), the constants pi and E, and the functions sqrt, "
            "exp and log"
        )


def _segment(src: str, node: ast.AST) -> str | None:
    """Source text of ``node`` (``None`` if the node has no position).

    The normalised formula is one line of ASCII characters, so the column
    offsets of the syntax tree index the string directly; this costs the
    length of the segment, while :func:`ast.get_source_segment` splits the
    whole source into lines on every call.
    """
    start = getattr(node, "col_offset", None)
    end = getattr(node, "end_col_offset", None)
    if start is None or end is None:
        return None
    if getattr(node, "lineno", 1) != 1 or getattr(node, "end_lineno", 1) != 1:
        return ast.get_source_segment(src, node)  # pragma: no cover - one line
    return src[start:end]


def _literal_text(src: str, node: ast.Constant) -> str:
    """Spelling of a numeric literal without digit-group underscores."""
    segment = _segment(src, node)
    return (segment if segment is not None else repr(node.value)).replace("_", "")


def _called_name(node: ast.AST) -> str | None:
    """Return the name a call node calls (``None`` for other nodes)."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _chain_family(node: ast.AST) -> str | None:
    """``"sum"`` for ``+ -``, ``"product"`` for ``* /``, else ``None``.

    A left-deep chain ``a + b - c + ...`` of one family counts as one
    nesting level and is built without recursion, so a long sum such
    as the partial fractions of a many-pole fit is not limited by
    :data:`MAX_DEPTH`.
    """
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return "sum"
        if isinstance(node.op, (ast.Mult, ast.Div)):
            return "product"
    return None


def _node_label(node: ast.AST) -> str:
    """Human-readable name of a syntax element."""
    if isinstance(node, (ast.operator, ast.unaryop)):
        return f"operator {type(node).__name__}"
    return f"syntax element {type(node).__name__}"


def _column(node: ast.AST) -> str:
    """`` at column N`` (1-based) when the node has a position."""
    col = getattr(node, "col_offset", None)
    return f" at column {col + 1}" if col is not None else ""


def _snippet(src: str, node: ast.AST, limit: int = 60) -> str:
    """Source text of ``node``, shortened to ``limit`` characters."""
    seg = _segment(src, node)
    seg = seg if seg is not None else src
    return repr(seg if len(seg) <= limit else seg[: limit - 3] + "...")


def _decimal_digits(n: int) -> int:
    """Return the number of decimal digits of a Python integer (``0`` has one)."""
    n = abs(n)
    if n < 10:
        return 1
    count = int(math.log10(n)) + 1  # may be one off next to a power of ten
    if n >= 10**count:
        return count + 1
    if n < 10 ** (count - 1):
        return count - 1
    return count


def _rational_digits(x: Fraction) -> int:
    """Return the digits of the numerator plus, for a fraction, the denominator."""
    digits = _decimal_digits(x.numerator)
    return digits if x.denominator == 1 else digits + _decimal_digits(x.denominator)


def _fraction_log10(x: Fraction) -> float:
    """``|log10|x||`` of a non-zero fraction (exact for big integers)."""
    return abs(math.log10(abs(x.numerator)) - math.log10(x.denominator))


def _decimal_log10(text: str) -> float | None:
    """``log10|x|`` of a decimal literal from its text; ``None`` for zero.

    The exponent is read as an integer and bounded before any float is
    formed: ``1e-<320 nines>`` is ``-inf``-like (``math.inf`` here),
    not ``0.0`` and not an :class:`OverflowError`.
    """
    match = _DECIMAL_LITERAL.fullmatch(text)
    if match is None:  # pragma: no cover - Python accepted the literal
        return math.inf
    whole, fraction = match["int"], match["frac"] or ""
    digits = (whole + fraction).lstrip("0")
    if not digits:
        return None
    exponent = (match["exp"] or "0").lstrip("+-").lstrip("0")
    if len(exponent) > 6:  # |exponent| >= 10**6: far outside every bound
        return math.inf
    shift = int(match["exp"] or "0") - len(fraction)
    return math.log10(int(digits)) + shift


class _Number(NamedTuple):
    """A number of the formula.

    ``value``: mpmath complex at :data:`_SHADOW_PREC` bits. ``exact``: the
    exact rational coefficient (``2*pi`` has 2; ``pi``, ``E`` and ``j``
    count as 1), ``None`` if the number is not exact (a float literal, a
    root or a function value is involved). ``rational``: the number *is*
    ``exact`` (no ``pi``, ``E`` or ``j`` factor).
    """

    value: Any
    exact: Fraction | None
    rational: bool


class _Part(NamedTuple):
    """A built sub-expression ``coefficient * factor_1 * factor_2 * ...``.

    ``factors``: unevaluated SymPy nodes that contain a free symbol (empty
    for a number). ``inner``: largest ``|log10|`` of a number inside the
    factors (the coefficient excluded). ``degree``: product of the
    exponents of nested powers (1 without powers).
    """

    coefficient: _Number
    factors: tuple[sp.Expr, ...]
    inner: float
    degree: float


class _Accumulator:
    """Running sum of the numbers that belong to one term of a sum."""

    __slots__ = ("value", "largest", "exact", "rational", "count", "factors", "inner")

    def __init__(self, factors: tuple[sp.Expr, ...], zero: Any) -> None:
        self.value = zero
        self.largest = zero.real
        self.exact: Fraction | None = Fraction(0)
        self.rational = True
        self.count = 0
        self.factors = factors
        self.inner = 0.0


class _Builder:
    """Build an unevaluated SymPy expression from an allow-listed syntax tree.

    Numbers are computed here, in :mod:`mpmath` at :data:`_SHADOW_PREC`
    bits, and bounded at every step; SymPy only stores nodes
    (``evaluate=False``). Every step costs bookkeeping proportional to the
    arguments it touches; the total is capped by :data:`MAX_WORK`.
    """

    def __init__(self, src: str, symbols: dict[str, sp.Symbol]) -> None:
        self._src = src
        self._symbols = symbols
        self._work = 0
        ctx = mpmath.MPContext()
        ctx.prec = _SHADOW_PREC
        self._ctx = ctx
        self._noise = ctx.ldexp(1, -_ZERO_SUM_BITS)
        self._zero = _Number(ctx.mpc(0), Fraction(0), True)
        self._one = _Number(ctx.mpc(1), Fraction(1), True)
        unit = _Number(ctx.mpc(0, 1), Fraction(1), False)
        self._constants = {
            "j": unit,
            "I": unit,
            "pi": _Number(ctx.mpc(+ctx.pi), Fraction(1), False),
            "E": _Number(ctx.mpc(+ctx.e), Fraction(1), False),
        }

    def _fail(self, message: str, node: ast.AST) -> FormulaError:
        return FormulaError(f"{message}{_column(node)}: {_snippet(self._src, node)}")

    def _spend(self, units: int, node: ast.AST) -> None:
        """Count bookkeeping work; reject the formula beyond :data:`MAX_WORK`."""
        self._work += units
        if self._work > MAX_WORK:
            raise self._fail(
                f"formula needs more than {MAX_WORK} bookkeeping steps to check", node
            )

    # ------------------------------------------------------------------
    # numbers
    # ------------------------------------------------------------------

    def _log10_size(self, value: Any) -> float:
        """``|log10|value||``; 0 for zero, ``inf`` if not finite."""
        if value == 0:
            return 0.0
        magnitude = abs(value)
        if not self._ctx.isfinite(magnitude):
            return math.inf
        return abs(float(self._ctx.log10(magnitude)))

    def _from_fraction(self, x: Fraction) -> Any:
        """The value of an exact fraction, correctly rounded."""
        mpf = self._ctx.make_mpf(
            from_rational(x.numerator, x.denominator, _SHADOW_PREC, round_nearest)
        )
        return self._ctx.mpc(mpf)

    def _number(self, value: Any, exact: Fraction | None, rational: bool) -> _Number:
        """A number whose value comes from ``exact`` when it is rational."""
        if rational and exact is not None:
            value = self._from_fraction(exact)
        elif value == 0:
            return self._zero
        return _Number(value, exact, rational)

    def _checked(self, number: _Number, node: ast.AST) -> _Number:
        """Reject a number outside the magnitude or digit bounds."""
        too_large = self._log10_size(number.value) > MAX_LOG10_MAGNITUDE
        exact = number.exact
        if exact is not None and exact != 0:
            too_large = too_large or _fraction_log10(exact) > MAX_LOG10_MAGNITUDE
        if too_large:
            raise self._fail(
                "number outside the bound "
                f"1e-{MAX_LOG10_MAGNITUDE:g} <= |x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if exact is not None and exact.denominator != 1:
            digits = max(
                _decimal_digits(exact.numerator), _decimal_digits(exact.denominator)
            )
            if digits > MAX_RATIONAL_DIGITS:
                raise self._fail(
                    f"fraction with more than {MAX_RATIONAL_DIGITS} digits", node
                )
        return number

    def _multiply(self, a: _Number, b: _Number, node: ast.AST) -> _Number:
        if a.value == 0 or b.value == 0:
            return self._zero
        both = a.exact is not None and b.exact is not None
        exact = a.exact * b.exact if both else None
        product = self._number(a.value * b.value, exact, a.rational and b.rational)
        return self._checked(product, node)

    def _divide(self, a: _Number, b: _Number, node: ast.AST) -> _Number:
        if b.value == 0:
            raise self._fail("division by zero", node)
        if a.value == 0:
            return self._zero
        both = a.exact is not None and b.exact is not None
        exact = a.exact / b.exact if both else None
        quotient = self._number(a.value / b.value, exact, a.rational and b.rational)
        return self._checked(quotient, node)

    def _negate(self, a: _Number) -> _Number:
        exact = None if a.exact is None else -a.exact
        return _Number(-a.value, exact, a.rational)

    def _integer_power(self, a: _Number, n: int, node: ast.AST) -> _Number:
        if a.value == 0:
            if n < 0:
                raise self._fail("zero raised to a negative power", node)
            return self._one if n == 0 else self._zero
        if n == 0:
            return self._one
        scale = self._log10_size(a.value)
        if a.exact is not None and a.exact != 0:
            scale = max(scale, _fraction_log10(a.exact))
        if abs(n) * scale > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "power would create a number outside the bound "
                f"|x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        exact = a.exact
        if exact is not None:
            digits = max(
                _decimal_digits(exact.numerator), _decimal_digits(exact.denominator)
            )
            if exact.denominator != 1 and abs(n) * digits > MAX_RATIONAL_DIGITS:
                raise self._fail(
                    "power would create a fraction with more than "
                    f"{MAX_RATIONAL_DIGITS} digits",
                    node,
                )
            exact = exact**n
        power = self._number(a.value**n, exact, a.rational)
        return self._checked(power, node)

    def _root_limits(self, exact: Fraction | None, q: int, node: ast.AST) -> None:
        """Limits for an exact number under a fractional power with denominator q."""
        if exact is None or exact in (0, 1, -1):
            return
        digits = _rational_digits(exact)
        if q > MAX_ROOT_DENOMINATOR or digits * q > MAX_RADICAND_DIGITS:
            raise self._fail(
                f"fractional power of an exact number with {digits} digits "
                f"and denominator {q} in the exponent (limits: denominator "
                f"<= {MAX_ROOT_DENOMINATOR}, digits x denominator <= "
                f"{MAX_RADICAND_DIGITS}); write the number as a float, "
                "e.g. 2.0 instead of 2, to evaluate the power numerically",
                node,
            )

    def _real_power(
        self, a: _Number, exponent: Fraction | float, node: ast.AST
    ) -> _Number:
        """``a**exponent`` for a non-integer exponent, on the principal branch."""
        if isinstance(exponent, Fraction):
            self._root_limits(a.exact, exponent.denominator, node)
            power = self._ctx.mpf(exponent.numerator) / exponent.denominator
        else:
            power = self._ctx.mpf(exponent)
        if a.value == 0:
            if exponent < 0:
                raise self._fail("zero raised to a negative power", node)
            return self._zero
        if abs(float(exponent)) * self._log10_size(a.value) > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "power would create a number outside the bound "
                f"|x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        value = self._ctx.power(a.value, power)
        return self._checked(self._number(value, None, False), node)

    def _function(self, name: str, a: _Number, node: ast.AST) -> _Number:
        """``exp`` or ``log`` of a number."""
        if name == "exp":
            if abs(a.value) > _MAX_EXP_ARGUMENT:
                raise self._fail(
                    f"exp argument larger than {_MAX_EXP_ARGUMENT:.1f} in magnitude",
                    node,
                )
            value = self._ctx.exp(a.value)
        else:
            if a.value == 0:
                raise self._fail("log(0) is undefined", node)
            value = self._ctx.log(a.value)
        return self._checked(self._number(value, None, False), node)

    # ------------------------------------------------------------------
    # SymPy nodes (never evaluated)
    # ------------------------------------------------------------------

    def _numeric(self, value: Any) -> sp.Expr:
        """Return an mpmath value as SymPy floats of 17 significant digits.

        A real or imaginary part below ``2**-100`` times the magnitude is
        rounding noise (``(-4)**(1/2)`` has the real part ``2*cos(pi/2)``)
        and is dropped.
        """
        ctx = self._ctx
        noise = abs(value) * self._noise
        real, imag = ctx.re(value), ctx.im(value)
        has_real, has_imag = abs(real) > noise, abs(imag) > noise
        if not (has_real or has_imag):
            return sp.S.Zero
        if not has_imag:
            return sp.Float(float(real), _NUMERIC_DIGITS)
        if imag == 1:
            imaginary = sp.I
        elif imag == -1:
            imaginary = sp.Mul(sp.S.NegativeOne, sp.I, evaluate=False)
        else:
            imaginary = sp.Mul(
                sp.Float(float(imag), _NUMERIC_DIGITS), sp.I, evaluate=False
            )
        if not has_real:
            return imaginary
        return sp.Add(sp.Float(float(real), _NUMERIC_DIGITS), imaginary, evaluate=False)

    def _coefficient_expression(self, value: Any) -> sp.Expr | None:
        """The coefficient of a product as a SymPy factor (``None`` for 1)."""
        if value == 1:
            return None
        if value == -1:
            return sp.S.NegativeOne
        if value == self._ctx.mpc(0, -1):
            return sp.Mul(sp.S.NegativeOne, sp.I, evaluate=False)
        return self._numeric(value)

    def expression(self, part: _Part) -> sp.Expr:
        """The unevaluated SymPy expression of a built part."""
        if not part.factors:
            return self._numeric(part.coefficient.value)
        items = list(part.factors)
        coefficient = self._coefficient_expression(part.coefficient.value)
        if coefficient is not None:
            items.insert(0, coefficient)
        if len(items) == 1:
            return items[0]
        return sp.Mul(*items, evaluate=False)

    def _scale(self, part: _Part) -> float:
        """Largest ``|log10|`` of a number in the part (coefficient included)."""
        return max(self._log10_size(part.coefficient.value), part.inner)

    @staticmethod
    def _inverse(factor: sp.Expr) -> sp.Expr:
        """``1/factor`` as a power (``(b**e)**-1`` is ``b**-e``)."""
        if isinstance(factor, sp.Pow):
            return sp.Pow(factor.base, -factor.exp, evaluate=False)
        return sp.Pow(factor, sp.S.NegativeOne, evaluate=False)

    def _constant(self, number: _Number, degree: float = 1.0) -> _Part:
        return _Part(number, (), 0.0, degree)

    # ------------------------------------------------------------------
    # syntax tree -> parts
    # ------------------------------------------------------------------

    def build(self, node: ast.AST) -> _Part:
        self._spend(1, node)
        if isinstance(node, ast.Constant):
            return self._constant(self._literal(node))
        if isinstance(node, ast.Name):
            if node.id in self._symbols:
                return _Part(self._one, (self._symbols[node.id],), 0.0, 1.0)
            if node.id in self._constants:
                return self._constant(self._constants[node.id])
            raise self._fail(
                f"function {node.id!r} must be called with one argument", node
            )
        if isinstance(node, ast.UnaryOp):
            operand = self.build(node.operand)
            if isinstance(node.op, ast.USub):
                return operand._replace(coefficient=self._negate(operand.coefficient))
            return operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            return self._power(node)
        if _chain_family(node) == "sum":
            return self._sum(node)
        if _chain_family(node) == "product":
            return self._product(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise self._fail(
            f"{_node_label(node)} is not allowed", node
        )  # pragma: no cover - _check_tree ran first

    @staticmethod
    def _chain(node: ast.BinOp) -> tuple[ast.AST, list[ast.BinOp]]:
        """Head operand and the links of a left-deep chain of one family."""
        family = _chain_family(node)
        links: list[ast.BinOp] = []
        while _chain_family(node) == family:
            links.append(node)
            node = node.left
        return node, links[::-1]

    def _literal(self, node: ast.Constant) -> _Number:
        """A numeric literal, bounded by its exact decimal value."""
        value = node.value
        text = _literal_text(self._src, node)
        ctx = self._ctx
        if type(value) is int:
            number = self._number(None, Fraction(value), True)
            return self._checked(number, node)
        if type(value) is float or type(value) is complex:
            imaginary = type(value) is complex
            digits = text[:-1] if imaginary else text  # an imaginary literal ends in j
            if not math.isfinite(value.imag if imaginary else value):
                raise self._fail("numeric literal is not finite", node)
            size = _decimal_log10(digits)
            if size is not None and abs(size) > MAX_LOG10_MAGNITUDE:
                raise self._fail(
                    "number outside the bound "
                    f"1e-{MAX_LOG10_MAGNITUDE:g} <= |x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                    node,
                )
            if size is None:
                return self._zero
            if imaginary and digits.isdigit():  # 2j: an exact coefficient times j
                exact = Fraction(int(digits))
                return _Number(
                    ctx.mpc(0, self._from_fraction(exact).real), exact, False
                )
            part = ctx.mpf(digits)
            return _Number(
                ctx.mpc(0, part) if imaginary else ctx.mpc(part), None, False
            )
        raise self._fail(f"literal of type {type(value).__name__} is not allowed", node)

    def _sum(self, node: ast.BinOp) -> _Part:
        """``a + b - c + ...``: numbers added, like terms collected."""
        head, links = self._chain(node)
        zero = self._zero.value
        terms: dict[tuple[sp.Expr, ...], _Accumulator] = {}
        common = 1  # least common multiple of the exact denominators
        degree = 1.0
        operands = [(head, False, node)] + [
            (link.right, isinstance(link.op, ast.Sub), link) for link in links
        ]
        for operand, subtract, where in operands:
            part = self.build(operand)
            self._spend(1 + len(part.factors), where)
            number = self._negate(part.coefficient) if subtract else part.coefficient
            degree = max(degree, part.degree)
            if number.exact is not None and number.exact.denominator != 1:
                q = number.exact.denominator
                common = common // math.gcd(common, q) * q
                if _decimal_digits(common) > MAX_RATIONAL_DIGITS:
                    raise self._fail(
                        f"fraction with more than {MAX_RATIONAL_DIGITS} digits", where
                    )
            if number.value == 0:
                continue
            term = terms.get(part.factors)
            if term is None:
                term = terms[part.factors] = _Accumulator(part.factors, zero)
            term.value += number.value
            term.largest = max(term.largest, abs(number.value))
            term.count += 1
            term.inner = max(term.inner, part.inner)
            if term.count == 1:
                term.exact, term.rational = number.exact, number.rational
            elif term.rational and number.rational:
                term.exact = term.exact + number.exact
            else:
                term.exact, term.rational = None, False
        constant = self._zero
        symbolic: list[_Part] = []
        for factors, term in terms.items():
            number = self._collected(term, node)
            if number is None:
                continue
            if factors:
                symbolic.append(_Part(number, factors, term.inner, 1.0))
            else:
                constant = number
        if not symbolic:
            return self._constant(constant, degree)
        if len(symbolic) == 1 and constant.value == 0:
            return symbolic[0]._replace(degree=degree)
        args = [self.expression(part) for part in symbolic]
        if constant.value != 0:
            args.append(self._numeric(constant.value))
        self._spend(len(args), node)
        inner = max(
            [self._scale(part) for part in symbolic]
            + [self._log10_size(constant.value)]
        )
        return _Part(self._one, (sp.Add(*args, evaluate=False),), inner, degree)

    def _collected(self, term: _Accumulator, node: ast.AST) -> _Number | None:
        """The number of one collected term; ``None`` if the term cancels."""
        if term.count == 0:
            return None
        if term.rational and term.exact is not None:
            if term.exact == 0:
                return None
            return self._checked(self._number(None, term.exact, True), node)
        if term.count > 1 and abs(term.value) <= term.largest * self._noise:
            return None  # cancels within the working precision
        if term.value == 0:
            return None
        return self._checked(_Number(term.value, term.exact, term.rational), node)

    def _product(self, node: ast.BinOp) -> _Part:
        """``a * b / c * ...``: numeric factors in one coefficient, bounded per step."""
        head, links = self._chain(node)
        part = self.build(head)
        coefficient = part.coefficient
        factors = list(part.factors)
        inner, degree = part.inner, part.degree
        for link in links:
            right = self.build(link.right)
            self._spend(1 + len(right.factors), link)
            if isinstance(link.op, ast.Div):
                coefficient = self._divide(coefficient, right.coefficient, link)
                factors.extend(self._inverse(factor) for factor in right.factors)
            else:
                coefficient = self._multiply(coefficient, right.coefficient, link)
                factors.extend(right.factors)
            inner, degree = max(inner, right.inner), max(degree, right.degree)
        if coefficient.value == 0:
            return self._constant(self._zero, degree)
        return _Part(coefficient, tuple(factors), inner, degree)

    @staticmethod
    def _signed_int(node: ast.AST) -> int | None:
        """``[+-]int`` literal as a Python integer, else ``None``."""
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sign * node.value
        return None

    def _exponent(self, node: ast.AST) -> Fraction | float | None:
        """Literal exponent ``[+-]int``, ``[+-]float`` or ``[+-]int/int``.

        Python reads ``(-3/2)`` as ``(-3)/2``; the sign may stand before
        the fraction or before its numerator. An integral float becomes an
        integer exponent.
        """
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(sign * node.value)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            if not math.isfinite(node.value):
                return None
            size = _decimal_log10(_literal_text(self._src, node))
            if size is not None and abs(size) > MAX_LOG10_MAGNITUDE:
                raise self._fail(
                    "exponent outside the bound "
                    f"1e-{MAX_LOG10_MAGNITUDE:g} <= |n| <= {MAX_POW}",
                    node,
                )
            value = sign * node.value
            return Fraction(int(value)) if value.is_integer() else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator = self._signed_int(node.left)
            denominator = node.right
            if (
                numerator is not None
                and isinstance(denominator, ast.Constant)
                and type(denominator.value) is int
                and denominator.value != 0
            ):
                exponent = Fraction(sign * numerator, denominator.value)
                if exponent != 0 and _fraction_log10(exponent) > MAX_LOG10_MAGNITUDE:
                    raise self._fail(
                        "exponent outside the bound "
                        f"1e-{MAX_LOG10_MAGNITUDE:g} <= |n| <= {MAX_POW}",
                        node,
                    )
                return exponent
        return None

    @staticmethod
    def _exponent_expression(exponent: Fraction | float) -> sp.Expr:
        if isinstance(exponent, Fraction):
            if exponent.denominator == 1:
                return sp.Integer(exponent.numerator)
            return sp.Rational(exponent.numerator, exponent.denominator)
        return sp.Float(exponent)

    def _power(self, node: ast.BinOp) -> _Part:
        exponent = self._exponent(node.right)
        if exponent is None or abs(exponent) > MAX_POW:
            raise self._fail(
                f"exponent must be a numeric literal with |n| <= {MAX_POW}", node
            )
        return self._raise(self.build(node.left), exponent, node)

    def _raise(self, base: _Part, exponent: Fraction | float, node: ast.AST) -> _Part:
        """``base**exponent``: bounds first, then a number or an unevaluated power."""
        size = abs(float(exponent))
        degree = size * base.degree
        if degree > MAX_POW:
            raise self._fail(
                f"power would create the exponent {degree:g} through nested "
                "powers (the exponents of nested powers multiply, also those "
                f"of numbers such as 10**6; limit {MAX_POW}); write such "
                "numbers as float literals, e.g. 1e6",
                node,
            )
        integral = isinstance(exponent, Fraction) and exponent.denominator == 1
        if not base.factors:
            number = base.coefficient
            if integral:
                number = self._integer_power(number, int(exponent), node)
            else:
                number = self._real_power(number, exponent, node)
            return self._constant(number, degree)
        if exponent == 0:
            return self._constant(self._one, degree)
        scale = self._scale(base)
        if size * scale > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "power would create a number outside the bound "
                f"|x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if isinstance(exponent, Fraction) and not integral:
            self._root_limits(base.coefficient.exact, exponent.denominator, node)
        if exponent == 1:
            return base._replace(degree=degree)
        self._spend(2, node)
        power = sp.Pow(
            self.expression(base), self._exponent_expression(exponent), evaluate=False
        )
        return _Part(self._one, (power,), size * scale, degree)

    def _call(self, node: ast.Call) -> _Part:
        func = node.func
        if not (
            isinstance(func, ast.Name)
            and func.id in _FUNCTIONS
            and len(node.args) == 1
            and not node.keywords
            and not isinstance(node.args[0], ast.Starred)
        ):
            raise self._fail(
                "only sqrt(x), exp(x) and log(x) with one positional argument "
                "may be called",
                node,
            )
        arg = self.build(node.args[0])
        if func.id == "sqrt":
            return self._raise(arg, Fraction(1, 2), node)
        if not arg.factors:
            return self._constant(
                self._function(func.id, arg.coefficient, node), arg.degree
            )
        self._spend(2, node)
        function = sp.exp if func.id == "exp" else sp.log
        call = function(self.expression(arg), evaluate=False)
        return _Part(self._one, (call,), self._scale(arg), arg.degree)


# ---------------------------------------------------------------------
# Numeric evaluation
# ---------------------------------------------------------------------


def formula_function(
    expr: sp.Expr,
    symbols: Iterable[str] = ("f", "rho"),
) -> Callable[..., np.ndarray]:
    r"""Return a NumPy function of an expression from :func:`parse_impedance_formula`.

    The expression tree is translated node by node into NumPy operations;
    SymPy is not asked to print, simplify or evaluate anything, so the
    cost is linear in the size of the expression (``sympy.lambdify``
    prints the expression, and its printer negates and reorders
    sub-expressions with SymPy's evaluation switched on).

    Parameters
    ----------
    expr : sympy.Expr
        Expression made of ``Add``, ``Mul``, ``Pow`` with a numeric
        exponent, ``exp``, ``log``, numbers, ``I``, ``pi``, ``E`` and
        symbols named in ``symbols``, as :func:`parse_impedance_formula`
        returns it.
    symbols : iterable of str, optional
        Names of the arguments of the returned function, in order
        (default ``f`` and ``rho``: frequency in Hz and soil resistivity
        in $\Omega\,\mathrm{m}$ by the ``groundinsight`` convention).

    Returns
    -------
    callable
        ``function(*values) -> numpy.ndarray`` of complex values,
        broadcast over the arguments; a constant expression gives a 0-d
        array. Real values are computed in real arithmetic (``float64``,
        as ``lambdify`` with NumPy does: ``1/(1 + 1/f)`` is 0 at
        ``f = 0``); a fractional power or a logarithm of a negative
        number takes SymPy's principal branch in complex arithmetic,
        where real arithmetic gives ``nan``.

    Raises
    ------
    ValueError
        If ``symbols`` is invalid, or the expression contains a node type
        outside the list above or a symbol that is not in ``symbols``.
    """
    names = _check_symbol_names(symbols)
    position = {name: index for index, name in enumerate(names)}
    program = _compile(expr, position)

    def function(*values: Any) -> np.ndarray:
        if len(values) != len(names):
            raise TypeError(
                f"the formula function takes {len(names)} arguments "
                f"({', '.join(names)}), got {len(values)}"
            )
        arrays = [_numeric_array(value) for value in values]
        return np.asarray(program(arrays), dtype=complex)

    return function


def _numeric_array(value: Any) -> np.ndarray:
    """``value`` as a float64 array, or complex128 if it has complex entries."""
    array = np.asarray(value)
    return array.astype(complex if np.iscomplexobj(array) else float)


def _real_exponent_power(base: Any, x: float) -> np.ndarray:
    """``base**x`` on the principal branch for a non-integer real ``x``.

    A base without negative or complex entries takes the real power
    (correctly rounded by the C library); the complex power
    ``exp(x*log(base))`` would lose about ``|x*log(base)|`` units in the
    last place (4e-14 relative at 1e295). Negative or complex entries take
    the complex power, positive entries of such a base the real one.
    """
    base = _numeric_array(base)
    if not np.iscomplexobj(base) and not np.any(base < 0.0):
        return np.power(base, x)
    base = base.astype(complex)
    positive = (base.imag == 0.0) & (base.real > 0.0)
    power = base**x
    if positive.any():
        real = np.power(np.where(positive, base.real, 1.0), x)
        power = np.where(positive, real, power)
    return power


def _logarithm(argument: Any) -> np.ndarray:
    """Natural logarithm on the principal branch (real for non-negative input)."""
    argument = _numeric_array(argument)
    if not np.iscomplexobj(argument) and not np.any(argument < 0.0):
        return np.log(argument)
    return np.log(argument.astype(complex))


def _compile(expr: sp.Basic, position: dict[str, int]) -> Callable[[list], Any]:
    """Translate an allow-listed expression tree into nested closures."""
    if isinstance(expr, sp.Symbol):
        if expr.name not in position:
            raise ValueError(
                f"the expression contains the symbol {expr.name!r}, which is not "
                f"one of the arguments {sorted(position)}"
            )
        index = position[expr.name]
        return lambda values: values[index]
    if expr is sp.I:
        return lambda values: 1j
    if isinstance(expr, (sp.Float, sp.Rational)) or expr in (sp.pi, sp.E):
        number = float(expr)
        return lambda values: number
    if isinstance(expr, (sp.Add, sp.Mul)):
        parts = [_compile(arg, position) for arg in expr.args]
        first, rest = parts[0], parts[1:]
        if isinstance(expr, sp.Add):

            def add(values: list) -> Any:
                total = first(values)
                for part in rest:
                    total = total + part(values)
                return total

            return add

        def multiply(values: list) -> Any:
            total = first(values)
            for part in rest:
                total = total * part(values)
            return total

        return multiply
    if isinstance(expr, sp.Pow):
        base = _compile(expr.base, position)
        exponent = expr.exp
        if exponent.is_Integer:
            n = int(exponent)
            return lambda values: _numeric_array(base(values)) ** n
        if isinstance(exponent, (sp.Rational, sp.Float)):
            x = float(exponent)
            return lambda values: _real_exponent_power(base(values), x)
        raise ValueError(f"cannot evaluate a power with the exponent {exponent!r}")
    if isinstance(expr, sp.exp):
        argument = _compile(expr.args[0], position)
        return lambda values: np.exp(_numeric_array(argument(values)))
    if isinstance(expr, sp.log) and len(expr.args) == 1:
        argument = _compile(expr.args[0], position)
        return lambda values: _logarithm(argument(values))
    raise ValueError(
        f"cannot evaluate a sub-expression of type {type(expr).__name__} "
        "(only Add, Mul, Pow, exp, log, numbers and symbols)"
    )


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def _significant_digits(significant_digits: object) -> int:
    """Validate a significant-digit count; values above 17 count as 17.

    Accepts integers (also NumPy's) and integral floats; rejects booleans,
    non-finite and fractional values and counts below 1.
    """
    if (
        isinstance(significant_digits, bool)
        or not isinstance(significant_digits, numbers.Real)
        or not math.isfinite(significant_digits)
        or not float(significant_digits).is_integer()
        or significant_digits < 1
    ):
        raise ValueError(
            "the number of significant digits must be a positive integer, "
            f"got {significant_digits!r}"
        )
    return min(int(significant_digits), _MAX_SIGNIFICANT_DIGITS)


def _rounded_text(value: float, digits: int) -> str:
    """``value`` correctly rounded to ``digits`` significant digits, as text.

    The one rounding rule of the formula renderer: Python's
    ``format(value, ".Ng")`` (identical on every platform), exponents
    without padding (``1e-7``), integral values without a decimal point.
    """
    text = format(value, f".{digits}g")
    mantissa, sep, exponent = text.partition("e")
    return f"{mantissa}e{int(exponent)}" if sep else text


def round_coefficients(expr: sp.Expr, significant_digits: int) -> sp.Expr:
    r"""Round every floating-point number of ``expr`` once.

    Parameters
    ----------
    expr : sympy.Expr
        Expression whose :class:`sympy.Float` atoms carry full float64
        precision (as built by the rho-f renderers).
    significant_digits : int
        Significant digits $N$ per coefficient, 1 to 17 (larger values
        count as 17).

    Returns
    -------
    sympy.Expr
        The same expression with every finite float replaced by its
        value correctly rounded to $N$ significant digits (the rule of
        :func:`format_impedance_formula`), stored as a SymPy float of $N$
        digits, so the expression prints compactly. A coefficient then
        carries a relative rounding error of at most
        $\tfrac12\,10^{1-N}$; integers, rationals, ``pi`` and ``E`` are
        left exact. Non-finite floats are left unchanged.

    Raises
    ------
    ValueError
        If ``significant_digits`` is not a positive integer.

    Notes
    -----
    Rounding happens once, from the full-precision value; printing the
    result with :func:`format_impedance_formula` at the same number of
    digits (up to 15) reproduces the text of the full-precision
    expression, because every decimal of at most 15 digits survives the
    conversion to float64.
    """
    digits = _significant_digits(significant_digits)
    mapping = {}
    for atom in expr.atoms(sp.Float):
        value = float(atom)
        if math.isfinite(value):
            mapping[atom] = sp.Float(_rounded_text(value, digits), digits)
    return expr.xreplace(mapping)


class _FormulaPrinter(StrPrinter):
    """String printer for the impedance-formula grammar.

    Floats are printed with a fixed number of significant digits using
    Python's correctly rounded ``format(x, ".Ng")`` (identical on every
    platform), the imaginary unit as ``j``.
    """

    def __init__(self, significant_digits: int) -> None:
        super().__init__()
        self._digits = significant_digits

    def _print_Float(self, expr: sp.Float) -> str:  # noqa: N802 (SymPy hook)
        value = float(expr)
        if not math.isfinite(value):
            raise ValueError(
                f"cannot render a non-finite coefficient ({value}) in an "
                "impedance formula"
            )
        return _rounded_text(value, self._digits)

    def _print_ImaginaryUnit(self, expr: sp.Expr) -> str:  # noqa: N802
        return "j"


def format_impedance_formula(expr: sp.Expr, *, significant_digits: int = 12) -> str:
    r"""Render a SymPy expression as an impedance-formula string.

    Parameters
    ----------
    expr : sympy.Expr
        Expression in the free symbols of the formula (e.g. ``f`` and
        ``rho``); the imaginary unit may appear as :data:`sympy.I` or
        as a symbol named ``j``.
    significant_digits : int, optional
        Significant digits of every floating-point coefficient, between
        1 and 17 (larger values are treated as 17, the precision of a
        float64). Default 12. Each coefficient is rounded once, so the
        relative error of a single coefficient is at most
        $\tfrac12\,10^{1-N}$ for $N$ significant digits; integers,
        rationals and the constants ``pi`` and ``E`` are printed
        exactly.

    Returns
    -------
    str
        Formula in the grammar accepted by :func:`parse_impedance_formula`,
        e.g. ``"41.5574270972 + 6820.15650671/(2*pi*f*j + 14825.095954)"``.
        Trailing zeros are dropped (``25`` instead of ``25.0000000000``),
        exponents are written without padding (``1e-7``).

    Raises
    ------
    ValueError
        If ``significant_digits`` is not a positive integer, or a
        coefficient is not finite.
    """
    digits = _significant_digits(significant_digits)
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError(
            "cannot render a non-finite coefficient (nan / oo) in an impedance formula"
        )
    return _FormulaPrinter(digits).doprint(expr)
