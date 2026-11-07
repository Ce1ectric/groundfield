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
digits per coefficient.

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
  (``2``, ``-1``, ``0.5``, ``(1/2)``, ``(-3/2)``) with
  $|n| \le$ :data:`MAX_POW`, so power towers such as ``9**9**9`` cannot
  be written;
* calls ``sqrt(x)``, ``exp(x)`` and ``log(x)`` with exactly one
  positional argument.

Resource bounds
---------------
* at most :data:`MAX_CHARS` characters, :data:`MAX_NODES` syntax-tree
  nodes and a nesting depth of :data:`MAX_DEPTH`;
* every numeric literal has at most :data:`MAX_LITERAL_CHARS`
  characters;
* every number inside the expression built so far, and every
  symbol-free sub-expression, is zero or lies within
  $10^{-300} \le |x| \le 10^{300}$ (:data:`MAX_LOG10_MAGNITUDE`); the
  numerator and denominator of an exact fraction have at most
  :data:`MAX_RATIONAL_DIGITS` digits;
* powers and ``exp`` are checked *before* they are computed, including
  powers of symbolic bases such as ``(10**60*f)**64``, whose numeric
  factor SymPy would raise to the power; so no huge integer is ever
  built;
* division by a numeric zero, ``0 ** -n`` and ``log(0)`` are rejected.

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
import string
import unicodedata
from collections.abc import Iterable

import sympy as sp
from sympy.printing.str import StrPrinter

__all__ = [
    "FormulaError",
    "parse_impedance_formula",
    "format_impedance_formula",
    "MAX_CHARS",
    "MAX_NODES",
    "MAX_DEPTH",
    "MAX_LITERAL_CHARS",
    "MAX_POW",
    "MAX_LOG10_MAGNITUDE",
    "MAX_RATIONAL_DIGITS",
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

MAX_LOG10_MAGNITUDE = 300.0
"""Bound on ``|log10|x||`` for every non-zero number and symbol-free value."""

MAX_RATIONAL_DIGITS = 1_000
"""Maximum digits of the numerator and denominator of an exact fraction.

Formulas written by the 0.15.0 exporter contain fractions with up to
254-digit numerators and denominators.
"""

_MAX_EXP_ARGUMENT = MAX_LOG10_MAGNITUDE * math.log(10.0)  # about 690.8
_MAX_SIGNIFICANT_DIGITS = 17  # float64 carries at most 17 significant digits

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
    or ``compile`` on the text (only :func:`ast.parse`). Every number in
    the expression built so far stays within the magnitude bound, and
    powers and exponentials are checked before they are evaluated, so
    the time and memory a formula costs, accepted or rejected, are
    bounded by the size limits.
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
    """Check size, depth, node types and names before anything is built."""
    allowed_names = set(names) | set(_CONSTANTS) | set(_FUNCTIONS)
    unknown: set[str] = set()
    n_nodes = 0
    stack: list[tuple[ast.AST, int]] = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
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
        family = _chain_family(node)
        for child in ast.iter_child_nodes(node):
            same_chain = family is not None and child is getattr(node, "left", None)
            same_chain = same_chain and _chain_family(child) == family
            stack.append((child, depth if same_chain else depth + 1))
    if unknown:
        raise FormulaError(
            f"formula contains unrecognised free symbols {sorted(unknown)!r}; "
            f"allowed symbols are {', '.join(names) or '(none)'}, the imaginary "
            "unit j (or I), the constants pi and E, and the functions sqrt, "
            "exp and log"
        )


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


def _value_log10(expr: sp.Expr) -> float | None:
    """``log10|expr|`` of a symbol-free expression (``-inf`` for zero).

    Returns ``None`` when ``expr`` has free symbols and ``inf`` when the
    value cannot be evaluated to a finite number. A non-zero value too
    small for a float64 returns ``-2 * MAX_LOG10_MAGNITUDE``.
    """
    if expr.free_symbols:
        return None
    try:
        value = complex(expr.evalf(20))
    except (OverflowError, TypeError, ValueError, ZeroDivisionError):
        return math.inf
    if not (math.isfinite(value.real) and math.isfinite(value.imag)):
        return math.inf
    magnitude = abs(value)
    if magnitude == 0.0:
        return -math.inf if expr.is_zero is not False else -2 * MAX_LOG10_MAGNITUDE
    return math.log10(magnitude)


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


def _number_sizes(expr: sp.Expr) -> tuple[float, float]:
    """Largest ``|log10|x||`` of the numbers in ``expr`` and largest
    numerator / denominator digit count of its fractions."""
    magnitude = 0.0
    fraction_digits = 0.0
    for atom in expr.atoms(sp.Number):
        if atom.is_zero:
            continue
        if isinstance(atom, sp.Integer):
            magnitude = max(magnitude, _digits(atom.p))
        elif isinstance(atom, sp.Rational):
            p, q = _digits(atom.p), _digits(atom.q)
            magnitude = max(magnitude, abs(p - q))
            fraction_digits = max(fraction_digits, p, q)
        elif isinstance(atom, sp.Float):
            magnitude = max(magnitude, abs(_float_log10(atom)))
        else:  # pragma: no cover - nan / oo are rejected separately
            magnitude = math.inf
    return magnitude, fraction_digits


class _Builder:
    """Build a SymPy expression from an allow-listed syntax tree."""

    def __init__(self, src: str, symbols: dict[str, sp.Symbol]) -> None:
        self._src = src
        self._symbols = symbols

    def _fail(self, message: str, node: ast.AST) -> FormulaError:
        return FormulaError(f"{message}{_column(node)}: {_snippet(self._src, node)}")

    def _bounded(self, expr: sp.Expr, node: ast.AST) -> sp.Expr:
        """Reject numbers or symbol-free values outside the bounds."""
        magnitude, fraction_digits = _number_sizes(expr)
        value = _value_log10(expr)
        if value is not None and value != -math.inf:
            magnitude = max(magnitude, abs(value))
        if magnitude > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "number outside the bound "
                f"1e-{MAX_LOG10_MAGNITUDE:g} <= |x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if fraction_digits > MAX_RATIONAL_DIGITS:
            raise self._fail(
                f"fraction with more than {MAX_RATIONAL_DIGITS} digits", node
            )
        return expr

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
        """``a + b - c + ...`` in one step (a sum cannot grow numbers)."""
        head, links = self._chain(node)
        terms = [self.build(head)]
        for link in links:
            term = self.build(link.right)
            terms.append(-term if isinstance(link.op, ast.Sub) else term)
        return self._bounded(sp.Add(*terms), node)

    def _product(self, node: ast.BinOp) -> sp.Expr:
        """``a * b / c * ...`` factor by factor, bounded after each step."""
        head, links = self._chain(node)
        result = self.build(head)
        for link in links:
            factor = self.build(link.right)
            if isinstance(link.op, ast.Div):
                if _value_log10(factor) == -math.inf:
                    raise self._fail("division by zero", link)
                result = result / factor
            else:
                result = result * factor
            result = self._bounded(result, link)
        return result

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
    def _exponent(node: ast.AST) -> sp.Number | None:
        """Literal exponent ``[+-]int``, ``[+-]float`` or ``[+-]int/int``."""
        sign = 1
        if isinstance(node, ast.UnaryOp):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            node = node.operand
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return sp.Integer(sign * node.value)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            if not math.isfinite(node.value):
                return None
            value = sign * node.value
            return sp.Integer(int(value)) if value.is_integer() else sp.Float(value)
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.right, ast.Constant)
            and type(node.left.value) is int
            and type(node.right.value) is int
            and node.right.value != 0
        ):
            return sp.Rational(sign * node.left.value, node.right.value)
        return None

    def _power(self, node: ast.BinOp) -> sp.Expr:
        exponent = self._exponent(node.right)
        if exponent is None or abs(exponent) > MAX_POW:
            raise self._fail(
                f"exponent must be a numeric literal with |n| <= {MAX_POW}", node
            )
        base = self.build(node.left)
        n = abs(float(exponent))
        value = _value_log10(base)
        if value == -math.inf and exponent < 0:
            raise self._fail("zero raised to a negative power", node)
        # SymPy raises every numeric factor of the base to the power, so
        # bound the result before it is computed.
        magnitude, fraction_digits = _number_sizes(base)
        if value is not None and value != -math.inf:
            magnitude = max(magnitude, abs(value))
        if n * magnitude > MAX_LOG10_MAGNITUDE:
            raise self._fail(
                "power would create a number outside the bound "
                f"|x| <= 1e{MAX_LOG10_MAGNITUDE:g}",
                node,
            )
        if n * fraction_digits > MAX_RATIONAL_DIGITS:
            raise self._fail(
                f"power would create a fraction with more than "
                f"{MAX_RATIONAL_DIGITS} digits",
                node,
            )
        return self._bounded(base**exponent, node)

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
        value = _value_log10(arg)
        if (
            func.id == "exp"
            and value is not None
            and value > math.log10(_MAX_EXP_ARGUMENT)
        ):
            raise self._fail(
                f"exp argument larger than {_MAX_EXP_ARGUMENT:.1f} in magnitude", node
            )
        if func.id == "log" and value == -math.inf:
            raise self._fail("log(0) is undefined", node)
        return self._bounded(_FUNCTIONS[func.id](arg), node)


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


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
        text = format(value, f".{self._digits}g")
        mantissa, sep, exponent = text.partition("e")
        return f"{mantissa}e{int(exponent)}" if sep else text

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
        If ``significant_digits`` is smaller than 1, or a coefficient is
        not finite.
    """
    digits = int(significant_digits)
    if digits != significant_digits or digits < 1:
        raise ValueError(
            f"significant_digits must be a positive integer, got {significant_digits!r}"
        )
    digits = min(digits, _MAX_SIGNIFICANT_DIGITS)
    if expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        raise ValueError(
            "cannot render a non-finite coefficient (nan / oo) in an impedance formula"
        )
    return _FormulaPrinter(digits).doprint(expr)
