r"""Safe parsing and rendering of impedance-formula strings.

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
never hands text to ``sympify``, ``parse_expr``, ``eval`` or ``exec``.
:func:`parse_impedance_formula` parses the text with :func:`ast.parse`,
which executes nothing, checks the syntax tree against an allow-list
and **builds** the SymPy expression node by node.
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
  fraction or its numerator) with $|n| \le$ :data:`MAX_POW`, so power
  towers such as ``9**9**9`` cannot be written; a fractional exponent
  ``p/q`` (also ``sqrt``, ``q = 2``) applied to an *exact* number, i.e.
  to an integer or fraction base or to the rational coefficient ``c`` of
  a product base ``c*f*...``, needs $q \le$ :data:`MAX_ROOT_DENOMINATOR`
  and (digits of the number) $\times\, q \le$ :data:`MAX_RADICAND_DIGITS`
  (``12**(1/3)`` and ``(2*f)**(3/2)`` are fine, a 300-digit integer under
  ``**(63/64)`` is not; written as a float, ``2.0**(1/100)``, a number is
  not limited);
* calls ``sqrt(x)``, ``exp(x)`` and ``log(x)`` with exactly one
  positional argument; ``log`` may not appear inside the argument of
  ``exp``, because SymPy rewrites ``exp(c*log(x))`` as the power
  ``x**c``, whose exponent ``c`` is no literal.

Resource bounds
---------------
* at most :data:`MAX_CHARS` characters, :data:`MAX_NODES` syntax-tree
  nodes and a nesting depth of :data:`MAX_DEPTH`;
* every numeric literal has at most :data:`MAX_LITERAL_CHARS`
  characters;
* every number of the expression, and every symbol-free sub-expression
  of it, is zero or lies within $10^{-300} \le |x| \le 10^{300}$
  (:data:`MAX_LOG10_MAGNITUDE`); the numerator and denominator of an
  exact fraction have at most :data:`MAX_RATIONAL_DIGITS` digits;
* nothing large is computed before it is bounded: powers and ``exp``
  are checked first, including the symbol-free factor of a symbolic
  base that SymPy raises to the power (``(10**60*f)**64``,
  ``(exp(600)*f)**64``); before a sum is built, the least common
  multiple of the denominators of its fractions is bounded (adding
  fractions with distinct large denominators would otherwise build an
  ever larger common denominator); a product is checked after every
  factor;
* values are bounded with a numeric shadow evaluation of every
  symbol-free node (:mod:`mpmath`, 128 bits), computed once from the
  values of its children, never by re-evaluating a sub-tree; a
  symbol-free sum smaller than $2^{-100}$ times its largest term
  counts as zero;
* the bookkeeping of one parse is capped at :data:`MAX_WORK` steps, so
  the cost of a formula, accepted or rejected, is bounded independently
  of its structure;
* division by a numeric zero, ``0 ** -n`` and ``log(0)`` are rejected.

SymPy evaluates some functions of numbers eagerly, in time that the
bounds above do not limit: it simplifies a fractional power of an exact
number by factorising the number, keeping a radicand of up to ``q - 1``
times its digits (the 300-digit integer ``10**300 - 1`` under ``63/64``
gives a radicand of about 18 800 digits, and SymPy tests a 16 280-digit
factor of it for primality; ``18**(10**30/(10**30 + 1))`` needs
``2**(10**30)``; products merge such powers and add their exponents),
and its ``log`` of a complex number ``r + r*j`` takes the
square root of ``2*r**2``. Without the rules below, 15 strings of the
formula corpus ran for more than 20 s each. So

* no exact number is ever left under a fractional power: a fractional
  power of a symbol-free base is evaluated numerically, and the rational
  coefficient ``c`` of a product base is taken out as the float
  ``|c|**(p/q)`` (``(-c*f)**(p/q) = |c|**(p/q) * (-f)**(p/q)``); a sum
  with a float coefficient gets float coefficients throughout (SymPy
  before 1.14 takes the largest coefficient of such a sum out of the
  power); otherwise powers of symbols, sums and functions keep their
  exact form;
* ``log`` of a symbol-free argument other than a positive rational or a
  float is evaluated numerically (``log(2)`` stays exact);
* a number created this way is a float of 17 significant digits per real
  and imaginary part (every float64 prints exactly), on SymPy's principal
  branch, and parts below ``2**-100`` times the magnitude are dropped.

The resulting expression contains only ``Add``, ``Mul``, ``Pow``,
numbers, ``I``, ``pi``, ``E``, ``exp``, ``log`` and the caller's
symbols, so code that :func:`sympy.lambdify` generates from it contains
nothing else either. The formulas written by every ``groundfield``
exporter up to 0.15.0 (rational coefficients with up to 254-digit
integers) and by the current one stay inside this grammar.
"""

from __future__ import annotations

import ast
import math
import numbers
import string
import unicodedata
from collections.abc import Iterable
from typing import Any, NamedTuple

import mpmath
import sympy as sp
from sympy.printing.str import StrPrinter

__all__ = [
    "FormulaError",
    "parse_impedance_formula",
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
"""Maximum magnitude of a literal exponent in ``base ** n``."""

MAX_ROOT_DENOMINATOR = 64
"""Maximum denominator ``q`` of a fractional exponent ``p/q`` of an exact number.

Applies to an integer or fraction base and to the rational coefficient of
a product base (``sqrt`` has ``q = 2``); the same limit as groundinsight's
formula guard.
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
"""Bound on ``|log10|x||`` for every non-zero number and symbol-free value."""

MAX_RATIONAL_DIGITS = 1_000
"""Maximum digits of the numerator and denominator of an exact fraction.

Formulas written by the 0.15.0 exporter contain fractions with up to
254-digit numerators and denominators.
"""

MAX_WORK = 200_000
"""Maximum bookkeeping steps of one parse.

A step is one argument of a node whose value or size is recorded, or one
term of a sum whose common denominator is accumulated. The longest
exporter formulas need a few thousand; quadratic patterns, such as a
product of several hundred factors, are rejected instead of being worked
through.
"""

_SHADOW_PREC = 128
"""Binary precision of the numeric shadow evaluation of symbol-free nodes."""

_ZERO_SUM_BITS = 100
"""A symbol-free sum below ``2**-100`` times its largest term counts as zero."""

_BALANCED_PRODUCT_MIN = 32
"""Product chains with at least this many ``*`` / ``/`` are multiplied pairwise.

Shorter chains, which include every product an exporter writes, keep the
left-to-right order of the text (the parsed expression is unchanged)."""

_MAX_EXP_ARGUMENT = MAX_LOG10_MAGNITUDE * math.log(10.0)  # about 690.8
_MAX_SIGNIFICANT_DIGITS = 17  # float64 carries at most 17 significant digits
_NUMERIC_DIGITS = 17
"""Significant digits of a float the parser computes (prints every float64 exactly)."""

_CONSTANTS = {"j": sp.I, "I": sp.I, "pi": sp.pi, "E": sp.E}
_FUNCTIONS = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log}
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
    r"""Parse an impedance formula without executing it.

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
        Expression whose free symbols are plain :class:`sympy.Symbol`
        objects named after a subset of ``symbols``; ``j`` and ``I``
        become :data:`sympy.I`.

    Raises
    ------
    FormulaError
        If the text is not a string, violates the grammar or exceeds a
        resource bound (see the module docstring).
    ValueError
        If ``symbols`` contains an invalid or reserved name.

    Notes
    -----
    Parsing never calls ``sympify``, ``parse_expr``, ``eval``, ``exec``
    or ``compile`` on the text (only :func:`ast.parse`). Every number of
    the expression built so far stays within the magnitude and digit
    bounds; powers, exponentials and the common denominator of a sum are
    checked before they are computed; values come from a numeric shadow
    evaluation that visits every node once; and the bookkeeping is capped
    at :data:`MAX_WORK` steps. No exact number is left under a fractional
    power and ``log`` inside ``exp`` is rejected, so SymPy never
    factorises a number or rewrites ``exp(c*log(x))`` as a power. So the
    time and memory a formula costs, accepted or rejected, are bounded by
    the size limits (audit 1 of the calendar release found sums and
    products inside those limits that took 13 s to 105 s, the
    groundinsight stream rational powers of large integers that ran for
    more than 20 s).
    """
    names = _check_symbol_names(symbols)
    src = _normalise(text)
    tree = _parse_tree(src)
    _check_tree(tree, src, names)
    builder = _Builder(src, {name: sp.Symbol(name) for name in names})
    try:
        expr = builder.build(tree.body)
    except RecursionError:  # pragma: no cover - guarded by MAX_DEPTH
        raise FormulaError("formula is nested too deeply") from None
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise FormulaError("formula evaluates to an infinite or undefined value")
    return expr


def _check_symbol_names(symbols: Iterable[str]) -> tuple[str, ...]:
    """Validate the caller's symbol names (a programming error if wrong)."""
    if isinstance(symbols, str):
        symbols = (symbols,)
    names = tuple(symbols)
    for name in names:
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
            raise ValueError(f"invalid symbol name {name!r}")
    reserved = sorted(set(names) & (set(_CONSTANTS) | set(_FUNCTIONS)))
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
    except (ValueError, MemoryError, RecursionError) as exc:
        raise FormulaError(f"not a valid expression: {exc}") from None


def _check_tree(tree: ast.Expression, src: str, names: tuple[str, ...]) -> None:
    """Check size, depth, node types and names before anything is built.

    Also rejects ``log`` inside the argument of ``exp``: SymPy rewrites
    ``exp(c*log(x))`` as ``x**c`` while it builds ``exp``, with an exponent
    ``c`` that is no literal (``exp(log(2*f)*10**12)`` builds
    ``2**(10**12)``).
    """
    allowed_names = set(names) | set(_CONSTANTS) | set(_FUNCTIONS)
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
        called = _called_name(node)
        if called == "log" and inside_exp:
            raise FormulaError(
                f"log() inside exp() is not allowed{_column(node)}: "
                f"{_snippet(src, node)}; SymPy rewrites exp(c*log(x)) as the "
                "power x**c, whose exponent c is no literal. Write the power "
                "x**c with a literal exponent instead"
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
    seg = ast.get_source_segment(src, node) if hasattr(node, "lineno") else None
    seg = seg if seg is not None else src
    return repr(seg if len(seg) <= limit else seg[: limit - 3] + "...")


def _digits(n: int) -> float:
    """``log10|n|`` of a non-zero Python integer (exact for big integers)."""
    return math.log10(abs(n))


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


def _float_coefficients(total: sp.Expr) -> sp.Expr:
    """Return a sum with a float coefficient with all its coefficients as floats.

    SymPy before 1.14 takes the largest coefficient ``c`` of such a sum out
    of a power, ``(2.0*f + c)**e -> c**e * (2.0/c*f + 1)**e``, and would
    factorise an exact ``c`` under a fractional ``e``. Coefficients of
    magnitude one and sums without a float coefficient are left as they
    are.
    """
    terms = [term.as_coeff_Mul() for term in total.args]
    if not any(coefficient.is_Float for coefficient, _ in terms):
        return total
    return sp.Add(
        *(
            sp.Float(float(c), _NUMERIC_DIGITS) * rest
            if c.is_Rational and abs(c) != 1
            else c * rest
            for c, rest in terms
        )
    )


def _rational_digits(x: sp.Rational) -> int:
    """Return the digits of the numerator plus, for a fraction, the denominator."""
    digits = _decimal_digits(int(x.p))
    return digits if x.q == 1 else digits + _decimal_digits(int(x.q))


def _float_log10(atom: sp.Float) -> float:
    """``log10|x|`` of a non-zero SymPy float (``inf`` if not finite)."""
    value = float(atom)
    if math.isfinite(value) and value != 0.0:
        return math.log10(abs(value))
    _sign, mantissa, exponent, bits = atom._mpf_
    if not mantissa:  # mpmath special values: zero handled by the caller
        return math.inf
    # outside the float64 range: |x| = mantissa * 2**exponent
    return (exponent + bits) * math.log10(2.0)


def _number_profile(atom: sp.Expr) -> tuple[float, float]:
    """``(|log10|x||, fraction digits)`` of a SymPy number (``inf`` if not finite)."""
    if atom.is_zero:
        return 0.0, 0.0
    if atom.is_Integer:
        return _digits(int(atom.p)), 0.0
    if atom.is_Rational:
        p, q = _digits(int(atom.p)), _digits(int(atom.q))
        return abs(p - q), max(p, q)
    if atom.is_Float:
        return abs(_float_log10(atom)), 0.0
    return math.inf, 0.0  # nan, zoo, oo


class _Profile(NamedTuple):
    """Size bookkeeping of one SymPy node (cached per node by :class:`_Builder`).

    ``atoms``: largest ``|log10|`` of a number atom in the node; ``scale``:
    largest ``|log10|`` of a number atom or of the value of a non-zero
    symbol-free sub-expression; ``fraction_digits``: largest digit count
    of a numerator or denominator of a fraction in the node.
    """

    symbolic: bool
    atoms: float
    scale: float
    fraction_digits: float


class _Builder:
    """Build a SymPy expression from an allow-listed syntax tree.

    Bounds are kept with a numeric *shadow* evaluation: every symbol-free
    node gets an :mod:`mpmath` value of :data:`_SHADOW_PREC` bits,
    computed once from the cached values of its children (or, for a
    product, from the previous partial product and the new factor), never
    by evaluating a whole sub-tree again. Every step costs bookkeeping
    proportional to the arguments of the node it creates; the total is
    capped by :data:`MAX_WORK`.
    """

    def __init__(self, src: str, symbols: dict[str, sp.Symbol]) -> None:
        self._src = src
        self._symbols = symbols
        self._ctx = mpmath.MPContext()
        self._ctx.prec = _SHADOW_PREC
        self._values: dict[sp.Basic, Any] = {}
        self._profiles: dict[sp.Basic, _Profile] = {}
        self._work = 0

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
    # numeric shadow evaluation and size bookkeeping
    # ------------------------------------------------------------------

    def _value(self, expr: sp.Expr, node: ast.AST) -> Any:
        """Value of a symbol-free expression (mpmath ``mpf`` / ``mpc``)."""
        cached = self._values.get(expr)
        if cached is not None:
            return cached
        ctx = self._ctx
        self._spend(1 + len(expr.args), node)
        try:
            if expr.is_Integer:
                value = ctx.mpf(int(expr.p))
            elif expr.is_Rational:
                value = ctx.mpf(int(expr.p)) / int(expr.q)
            elif expr.is_Float:
                value = ctx.make_mpf(expr._mpf_)
            elif expr is sp.I:
                value = ctx.mpc(0, 1)
            elif expr is sp.pi:
                value = +ctx.pi
            elif expr is sp.E:
                value = +ctx.e
            elif expr.is_Add:
                terms = [self._value(arg, node) for arg in expr.args]
                value = ctx.fsum(terms)
                largest = max(abs(t) for t in terms)
                if abs(value) <= largest * ctx.ldexp(1, -_ZERO_SUM_BITS):
                    value = ctx.mpf(0)  # cancels within the working precision
            elif expr.is_Mul:
                value = ctx.fprod(self._value(arg, node) for arg in expr.args)
            elif expr.is_Pow:
                base = self._value(expr.base, node)
                if expr.exp.is_Integer:
                    value = base ** int(expr.exp)
                else:
                    value = ctx.power(base, self._value(expr.exp, node))
            elif isinstance(expr, sp.exp):
                value = ctx.exp(self._value(expr.args[0], node))
            elif isinstance(expr, sp.log):
                value = ctx.log(self._value(expr.args[0], node))
            elif expr.is_Number:  # nan, zoo, oo
                value = ctx.inf
            else:  # pragma: no cover - the grammar builds nothing else
                raise self._fail(
                    f"cannot bound a sub-expression of type {type(expr).__name__}", node
                )
        except (ZeroDivisionError, ValueError, OverflowError):
            value = ctx.inf
        self._values[expr] = value
        return value

    def _log10_size(self, value: Any) -> float:
        """``|log10|value||``; 0 for zero, ``inf`` if not finite."""
        if value == 0:
            return 0.0
        magnitude = abs(value)
        if not self._ctx.isfinite(magnitude):
            return math.inf
        return abs(float(self._ctx.log10(magnitude)))

    def _profile(self, expr: sp.Expr, node: ast.AST) -> _Profile:
        """Cached :class:`_Profile` of ``expr`` (computed from its arguments)."""
        cached = self._profiles.get(expr)
        if cached is not None:
            return cached
        self._spend(1 + len(expr.args), node)
        if expr.is_Symbol:
            profile = _Profile(True, 0.0, 0.0, 0.0)
        elif expr.is_Number:
            size, fraction = _number_profile(expr)
            profile = _Profile(False, size, size, fraction)
        else:
            parts = [self._profile(arg, node) for arg in expr.args]
            symbolic = any(p.symbolic for p in parts)
            atoms = max((p.atoms for p in parts), default=0.0)
            scale = max((p.scale for p in parts), default=0.0)
            fraction = max((p.fraction_digits for p in parts), default=0.0)
            if not symbolic:
                scale = max(scale, self._log10_size(self._value(expr, node)))
            profile = _Profile(symbolic, atoms, scale, fraction)
        self._profiles[expr] = profile
        return profile

    def _bounded(self, expr: sp.Expr, node: ast.AST) -> sp.Expr:
        """Reject numbers or symbol-free values outside the bounds."""
        profile = self._profile(expr, node)
        if profile.scale > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "number outside the bound "
                f"1e-{MAX_LOG10_MAGNITUDE:g} <= |x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if profile.fraction_digits > MAX_RATIONAL_DIGITS:
            raise self._fail(
                f"fraction with more than {MAX_RATIONAL_DIGITS} digits", node
            )
        return expr

    def _is_zero(self, expr: sp.Expr, node: ast.AST) -> bool:
        """Whether ``expr`` is a symbol-free expression of value zero."""
        return not self._profile(expr, node).symbolic and self._value(expr, node) == 0

    def _numeric(self, value: Any) -> sp.Expr:
        """Return an mpmath value as SymPy floats of 17 significant digits.

        A real or imaginary part below ``2**-100`` times the magnitude is
        rounding noise of the shadow evaluation (``(-4)**(1/2)`` has the
        real part ``2*cos(pi/2)``) and is dropped.
        """
        ctx = self._ctx
        magnitude = abs(value)
        noise = magnitude * ctx.ldexp(1, -_ZERO_SUM_BITS)
        result: sp.Expr = sp.Integer(0)
        real, imag = ctx.re(value), ctx.im(value)
        if abs(real) > noise:
            result += sp.Float(float(real), _NUMERIC_DIGITS)
        if abs(imag) > noise:
            result += sp.Float(float(imag), _NUMERIC_DIGITS) * sp.I
        return result

    # ------------------------------------------------------------------
    # syntax tree -> SymPy
    # ------------------------------------------------------------------

    def build(self, node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            return self._literal(node)
        if isinstance(node, ast.Name):
            if node.id in self._symbols:
                return self._symbols[node.id]
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise self._fail(
                f"function {node.id!r} must be called with one argument", node
            )
        if isinstance(node, ast.UnaryOp):
            operand = self.build(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
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
        )  # pragma: no cover

    @staticmethod
    def _chain(node: ast.BinOp) -> tuple[ast.AST, list[ast.BinOp]]:
        """Head operand and the links of a left-deep chain of one family."""
        family = _chain_family(node)
        links: list[ast.BinOp] = []
        while _chain_family(node) == family:
            links.append(node)
            node = node.left
        return node, links[::-1]

    def _sum(self, node: ast.BinOp) -> sp.Expr:
        """``a + b - c + ...`` in one step, its common denominator bounded first."""
        head, links = self._chain(node)
        terms = [self.build(head)]
        common = self._common_denominator(1, terms[0], node)
        for link in links:
            term = self.build(link.right)
            terms.append(-term if isinstance(link.op, ast.Sub) else term)
            common = self._common_denominator(common, term, link)
        return self._bounded(sp.Add(*terms), node)

    def _common_denominator(self, common: int, term: sp.Expr, node: ast.AST) -> int:
        """Fold the fraction denominators of ``term`` into ``common``.

        Adding fractions (numbers, or the coefficients of like terms)
        needs their least common multiple; with distinct large prime
        denominators it grows with every term (audit 1, H1: 222 terms of
        900 digits took more than a minute before the result was
        rejected). The least common multiple of the fraction coefficients
        of all terms bounds every denominator the sum can create; it is
        accumulated term by term, before the sum is built, and the formula
        is rejected as soon as it has more than :data:`MAX_RATIONAL_DIGITS`
        digits.
        """
        for item in term.args if term.is_Add else (term,):
            self._spend(1, node)
            coefficient = item.as_coeff_Mul()[0]
            if coefficient.is_Rational and coefficient.q != 1:
                q = int(coefficient.q)
                common = common // math.gcd(common, q) * q
                if _digits(common) > MAX_RATIONAL_DIGITS:
                    raise self._fail(
                        f"fraction with more than {MAX_RATIONAL_DIGITS} digits", node
                    )
        return common

    def _seed_product(
        self,
        product: sp.Expr,
        left: sp.Expr,
        right: sp.Expr,
        divide: bool,
        node: ast.AST,
    ) -> None:
        """Record the value of ``left * right`` (or ``left / right``) directly.

        Saves the re-evaluation of a growing symbol-free product (audit 1,
        H2: quadratic in the number of factors).
        """
        if (
            product not in self._values
            and not self._profile(left, node).symbolic
            and not self._profile(right, node).symbolic
        ):
            a, b = self._value(left, node), self._value(right, node)
            self._values[product] = a / b if divide else a * b

    def _product(self, node: ast.BinOp) -> sp.Expr:
        """``a * b / c * ...`` factor by factor, bounded after each step.

        The value of a symbol-free partial product is the previous value
        times (or over) the factor's value, so no step re-evaluates the
        product so far. Chains longer than :data:`_BALANCED_PRODUCT_MIN`
        factors are multiplied pairwise instead (:meth:`_balanced_product`).
        """
        head, links = self._chain(node)
        if len(links) >= _BALANCED_PRODUCT_MIN:
            return self._balanced_product(head, links, node)
        result = self.build(head)
        for link in links:
            factor = self.build(link.right)
            divide = isinstance(link.op, ast.Div)
            if divide and self._is_zero(factor, link):
                raise self._fail("division by zero", link)
            product = result / factor if divide else result * factor
            self._seed_product(product, result, factor, divide, link)
            result = self._bounded(product, link)
        return result

    def _balanced_product(
        self, head: ast.AST, links: list[ast.BinOp], node: ast.AST
    ) -> sp.Expr:
        """Multiply a long chain pairwise, bounding every partial product.

        SymPy sorts the arguments of a product whenever one is added, so
        multiplying ``n`` factors one at a time costs ``O(n**2 log n)``
        (audit 1, H2: 332 logarithms took 4.8 s after the bookkeeping was
        made linear); pairwise merging costs ``O(n log**2 n)`` and every
        merge combines two bounded operands.
        """
        factors = [self.build(head)]
        for link in links:
            factor = self.build(link.right)
            if isinstance(link.op, ast.Div):
                if self._is_zero(factor, link):
                    raise self._fail("division by zero", link)
                inverse = sp.Integer(1) / factor
                self._seed_product(inverse, sp.Integer(1), factor, True, link)
                factor = self._bounded(inverse, link)
            factors.append(factor)
        while len(factors) > 1:
            merged = []
            for left, right in zip(factors[0::2], factors[1::2], strict=False):
                product = left * right
                self._seed_product(product, left, right, False, node)
                merged.append(self._bounded(product, node))
            if len(factors) % 2:
                merged.append(factors[-1])
            factors = merged
        return factors[0]

    def _literal(self, node: ast.Constant) -> sp.Expr:
        value = node.value
        seg = (ast.get_source_segment(self._src, node) or repr(value)).replace("_", "")
        if len(seg) > MAX_LITERAL_CHARS:
            raise self._fail(
                f"numeric literal longer than {MAX_LITERAL_CHARS} characters", node
            )
        if type(value) is int:
            return self._bounded(sp.Integer(value), node)
        if type(value) is float:
            if not math.isfinite(value):
                raise self._fail("numeric literal is not finite", node)
            return self._bounded(sp.Float(seg), node)
        if type(value) is complex:
            if not math.isfinite(value.imag):
                raise self._fail("numeric literal is not finite", node)
            imag_text = seg[:-1]  # a Python imaginary literal ends in j or J
            imag = sp.Integer(imag_text) if imag_text.isdigit() else sp.Float(imag_text)
            return self._bounded(imag * sp.I, node)
        raise self._fail(f"literal of type {type(value).__name__} is not allowed", node)

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

    @classmethod
    def _exponent(cls, node: ast.AST) -> sp.Number | None:
        """Literal exponent ``[+-]int``, ``[+-]float`` or ``[+-]int/int``.

        Python reads ``(-3/2)`` as ``(-3)/2``; the sign may stand before
        the fraction or before its numerator.
        """
        sign = 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(sign * node.value)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            if not math.isfinite(node.value):
                return None
            value = sign * node.value
            return sp.Integer(int(value)) if value.is_integer() else sp.Float(value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            numerator = cls._signed_int(node.left)
            denominator = node.right
            if (
                numerator is not None
                and isinstance(denominator, ast.Constant)
                and type(denominator.value) is int
                and denominator.value != 0
            ):
                return sp.Rational(sign * numerator, denominator.value)
        return None

    def _power_scale(self, base: sp.Expr, profile: _Profile, node: ast.AST) -> float:
        """Largest ``|log10|`` the power multiplies by its exponent.

        The number atoms of the base, the value of a symbol-free base, and
        the symbol-free factor of a product base: SymPy distributes an
        integer power over a product, so ``(exp(600)*f)**64`` would create
        ``exp(38400)`` (audit 1, L1).
        """
        scale = profile.atoms
        if not profile.symbolic:
            return max(scale, self._log10_size(self._value(base, node)))
        if base.is_Mul:
            free = [arg for arg in base.args if not self._profile(arg, node).symbolic]
            if free:
                value = self._ctx.fprod(self._value(arg, node) for arg in free)
                scale = max(scale, self._log10_size(value))
        return scale

    def _power(self, node: ast.BinOp) -> sp.Expr:
        exponent = self._exponent(node.right)
        if exponent is None or abs(exponent) > MAX_POW:
            raise self._fail(
                f"exponent must be a numeric literal with |n| <= {MAX_POW}", node
            )
        base = self.build(node.left)
        n = abs(float(exponent))
        if exponent < 0 and self._is_zero(base, node):
            raise self._fail("zero raised to a negative power", node)
        # SymPy raises every numeric factor of the base to the power, so
        # bound the result before it is computed.
        profile = self._profile(base, node)
        if n * self._power_scale(base, profile, node) > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "power would create a number outside the bound "
                f"|x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if n * profile.fraction_digits > MAX_RATIONAL_DIGITS:
            raise self._fail(
                f"power would create a fraction with more than "
                f"{MAX_RATIONAL_DIGITS} digits",
                node,
            )
        if exponent.is_Rational and not exponent.is_Integer:
            return self._bounded(self._fractional_power(base, exponent, node), node)
        return self._bounded(base**exponent, node)

    def _fractional_power(
        self, base: sp.Expr, exponent: sp.Rational, node: ast.AST
    ) -> sp.Expr:
        """Return ``base**(p/q)`` for a non-integer ``p/q`` without an exact root.

        SymPy simplifies a fractional power of an exact number by
        factorising the number and keeps a radicand of up to ``q - 1``
        times its digits; a product merges such powers and adds their
        exponents (``12**(58/59)*12**(60/61)*...``). So a symbol-free base
        is powered numerically, and the rational coefficient ``c`` of a
        product base is taken out as the float ``|c|**(p/q)``, which is
        exact for the principal branch: ``(c*x)**e = c**e * x**e`` for
        ``c > 0``, and ``(c*x)**e = |c|**e * (-x)**e`` for ``c < 0``.
        Before that, an exact number must satisfy the limits
        :data:`MAX_ROOT_DENOMINATOR` and :data:`MAX_RADICAND_DIGITS`.
        """
        p, q = int(exponent.p), int(exponent.q)
        coefficient, rest = base.as_coeff_Mul()
        exact = coefficient.is_Rational and coefficient not in (0, 1, -1)
        if exact:
            digits = _rational_digits(coefficient)
            if q > MAX_ROOT_DENOMINATOR or digits * q > MAX_RADICAND_DIGITS:
                raise self._fail(
                    f"fractional power of an exact number with {digits} digits "
                    f"and denominator {q} in the exponent (limits: denominator "
                    f"<= {MAX_ROOT_DENOMINATOR}, digits x denominator <= "
                    f"{MAX_RADICAND_DIGITS}); write the number as a float, "
                    "e.g. 2.0 instead of 2, to evaluate the power numerically",
                    node,
                )
        ctx = self._ctx
        power = ctx.mpf(p) / q
        if not self._profile(base, node).symbolic:
            return self._numeric(ctx.power(self._value(base, node), power))
        if not exact:
            if base.is_Add:
                base = self._bounded(_float_coefficients(base), node)
            return base**exponent
        magnitude = abs(coefficient)
        factor = self._numeric(ctx.power(self._value(magnitude, node), power))
        return factor * (rest if coefficient > 0 else -rest) ** exponent

    def _call(self, node: ast.Call) -> sp.Expr:
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
            return self._bounded(self._fractional_power(arg, sp.S.Half, node), node)
        if not self._profile(arg, node).symbolic:
            value = self._value(arg, node)
            if func.id == "exp" and abs(value) > _MAX_EXP_ARGUMENT:
                raise self._fail(
                    f"exp argument larger than {_MAX_EXP_ARGUMENT:.1f} in magnitude",
                    node,
                )
            if func.id == "log" and value == 0:
                raise self._fail("log(0) is undefined", node)
            if func.id == "log" and not (
                arg.is_Float or (arg.is_Rational and arg.is_positive)
            ):
                # SymPy's log of r + r*j takes sqrt(2*r**2) and factorises it
                return self._bounded(self._numeric(self._ctx.log(value)), node)
        return self._bounded(_FUNCTIONS[func.id](arg), node)


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
