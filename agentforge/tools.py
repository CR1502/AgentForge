"""Tool definitions for the tool-calling agent.

CONTRACT (do not change without updating state.py + scenarios + eval):
    - Expose two LangChain tools whose `.name` are exactly:
        "calculator"      -> evaluate an arithmetic expression
        "unit_converter"  -> convert a value between supported units
    - Expose `TOOLS: list` containing both tool objects, in that order.

Implemented by the tools agent in Phase 1.

Self-contained: stdlib only + langchain_core.
"""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# calculator: AST-based safe arithmetic evaluator
# ---------------------------------------------------------------------------

# Allowed binary operators mapped to their safe implementations.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators.
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guard rail: refuse absurdly large exponents to avoid DoS via huge integers.
_MAX_POW_EXPONENT = 1000


def _safe_eval(node: ast.AST):
    """Recursively evaluate a parsed arithmetic AST node.

    Only numeric literals, parentheses, unary +/-, and the binary operators
    in ``_BIN_OPS`` are permitted. Any name, call, attribute access, or other
    construct raises ``ValueError``.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    # Numeric literals only (reject str/bytes/bool-as-name/None/etc.).
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"unsupported constant: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unsupported unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](_safe_eval(node.operand))

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a pure arithmetic expression and return the numeric result.

    Use this whenever the user asks you to compute a math expression: addition,
    subtraction, multiplication, division, integer/floor division, modulo (remainder),
    exponentiation, and nested parentheses with unary plus/minus. Operands may be
    integers or floats. This tool does NOT convert units — use `unit_converter` for
    that, and do NOT use it for anything other than arithmetic (no variables,
    function calls, or text).

    Supported operators: ``+  -  *  /  //  %  **`` and parentheses ``( )``.

    Examples:
        calculator("2 + 3 * 4")        -> "14"
        calculator("(1 + 2) ** 3")     -> "27"
        calculator("10 / 4")           -> "2.5"
        calculator("17 % 5")           -> "2"
        calculator("-(6 + 1) * 2")     -> "-14"

    Args:
        expression: The arithmetic expression to evaluate, as a string.

    Returns:
        The computed value rendered as a string, or an ``"Error: ..."`` message
        if the expression is empty, malformed, or contains unsupported constructs.
    """
    if not isinstance(expression, str) or not expression.strip():
        return "Error: empty expression"
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
    except ZeroDivisionError:
        return "Error: division by zero"
    except SyntaxError:
        return f"Error: could not parse expression: {expression!r}"
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return f"Error: {exc}"
    return str(result)


# ---------------------------------------------------------------------------
# unit_converter: fixed-set unit conversions across a few categories
# ---------------------------------------------------------------------------

# Linear-unit categories: value stored as a factor to the category base unit.
#   length base = metre; mass base = gram.
_LENGTH_TO_BASE = {  # -> metres
    "km": 1000.0,
    "m": 1.0,
    "cm": 0.01,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
}
_MASS_TO_BASE = {  # -> grams
    "kg": 1000.0,
    "g": 1.0,
    "lb": 453.59237,
    "oz": 28.349523125,
}

# Temperature unit aliases -> canonical key.
_TEMP_ALIASES = {
    "c": "c",
    "celsius": "c",
    "f": "f",
    "fahrenheit": "f",
    "k": "k",
    "kelvin": "k",
}

_ROUND = 4


def _convert_temperature(value: float, src: str, dst: str) -> float:
    # Normalise to Celsius first.
    if src == "c":
        celsius = value
    elif src == "f":
        celsius = (value - 32.0) * 5.0 / 9.0
    else:  # kelvin
        celsius = value - 273.15

    if dst == "c":
        return celsius
    if dst == "f":
        return celsius * 9.0 / 5.0 + 32.0
    return celsius + 273.15  # kelvin


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value from one unit of measure to another.

    Use this whenever the user asks to convert a quantity between units of the
    same kind (length, mass/weight, or temperature). Unit names are
    case-insensitive. This tool does NOT do general arithmetic — use
    `calculator` for math.

    Supported units by category (conversions only work within one category):
        length:       km, mi, m, ft, cm, in
        mass/weight:  kg, lb, g, oz
        temperature:  c / celsius, f / fahrenheit, k / kelvin

    Examples:
        unit_converter(10, "km", "mi")        -> "10 km = 6.2137 mi"
        unit_converter(5, "ft", "m")          -> "5 ft = 1.524 m"
        unit_converter(2.5, "kg", "lb")       -> "2.5 kg = 5.5116 lb"
        unit_converter(100, "celsius", "f")   -> "100 celsius = 212.0 f"
        unit_converter(1, "kg", "mi")         -> "Error: incompatible units ..."

    Args:
        value: The numeric amount to convert.
        from_unit: The unit the value is currently in.
        to_unit: The unit to convert the value into.

    Returns:
        A human-readable string of the form ``"<value> <from> = <result> <to>"``
        with the result rounded to 4 decimals, or an ``"Error: ..."`` message if a
        unit is unknown or the two units belong to different categories.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"Error: value must be numeric, got {value!r}"

    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        return "Error: unit names must be strings"

    src = from_unit.strip().lower()
    dst = to_unit.strip().lower()

    # Determine the category of each unit.
    def category(u: str) -> str | None:
        if u in _LENGTH_TO_BASE:
            return "length"
        if u in _MASS_TO_BASE:
            return "mass"
        if u in _TEMP_ALIASES:
            return "temperature"
        return None

    src_cat = category(src)
    dst_cat = category(dst)

    if src_cat is None:
        return f"Error: unknown unit {from_unit!r}"
    if dst_cat is None:
        return f"Error: unknown unit {to_unit!r}"
    if src_cat != dst_cat:
        return (
            f"Error: incompatible units {from_unit!r} ({src_cat}) and "
            f"{to_unit!r} ({dst_cat}); can only convert within the same category"
        )

    try:
        if src_cat == "temperature":
            result = _convert_temperature(value, _TEMP_ALIASES[src], _TEMP_ALIASES[dst])
        elif src_cat == "length":
            result = value * _LENGTH_TO_BASE[src] / _LENGTH_TO_BASE[dst]
        else:  # mass
            result = value * _MASS_TO_BASE[src] / _MASS_TO_BASE[dst]
    except Exception as exc:  # pragma: no cover - defensive catch-all
        return f"Error: {exc}"

    result = round(result, _ROUND)
    # Render value tidily: drop a trailing ".0" on whole-number inputs.
    value_str = f"{value:g}"
    return f"{value_str} {src} = {result} {dst}"


TOOLS: list = [calculator, unit_converter]


if __name__ == "__main__":
    print("== tool names ==")
    print([t.name for t in TOOLS])

    print("\n== calculator ==")
    for expr in ["2 + 3 * 4", "(1 + 2) ** 3", "10 / 4", "17 % 5", "-(6 + 1) * 2"]:
        print(f"  {expr!r:20} -> {calculator.invoke({'expression': expr})}")
    # Error cases
    print(f"  {'1/0':20} -> {calculator.invoke({'expression': '1/0'})}")
    print(f"  {'__import__(x)':20} -> {calculator.invoke({'expression': '__import__(\"os\")'})}")
    print(f"  {'2 + foo':20} -> {calculator.invoke({'expression': '2 + foo'})}")

    print("\n== unit_converter ==")
    conversions = [
        (10, "km", "mi"),
        (5, "ft", "m"),
        (2.5, "kg", "lb"),
        (100, "celsius", "f"),
        (0, "c", "k"),
        (16, "oz", "g"),
    ]
    for v, f, t in conversions:
        print(f"  {v} {f}->{t:8} -> {unit_converter.invoke({'value': v, 'from_unit': f, 'to_unit': t})}")
    # Error cases
    print(f"  incompatible      -> {unit_converter.invoke({'value': 1, 'from_unit': 'kg', 'to_unit': 'mi'})}")
    print(f"  unknown unit      -> {unit_converter.invoke({'value': 1, 'from_unit': 'furlong', 'to_unit': 'm'})}")
