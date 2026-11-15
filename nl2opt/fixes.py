"""Optional transforms applied to extracted code before it runs.

Each fix is a pure function code -> code, registered by name, so experiments can switch fixes on
one at a time and measure each separately (see nl2opt.rescore).
"""

import ast
import re

PULP_IMPORTS = "import pulp\nfrom pulp import *  # added by nl2opt fix 'imports'\n"


def add_pulp_imports(code):
    """Make every PuLP name available (value, LpStatus, LpMinimize, ...).

    Prepended, so the program's own imports and definitions still take precedence. Typos in the
    program's own variable names are deliberately not touched.
    """
    return PULP_IMPORTS + code


# Solver names a model might pass as a string -> PuLP's getSolver name. Matched case-insensitively.
SOLVER_ALIASES = {"highs": "HiGHS", "highs_cmd": "HiGHS_CMD", "cbc": "PULP_CBC_CMD", "pulp_cbc_cmd": "PULP_CBC_CMD",
                  "coin": "PULP_CBC_CMD", "glpk": "GLPK_CMD", "glpk_cmd": "GLPK_CMD"}
SOLVE_WITH_STRING = re.compile(
    r"\.solve\(\s*(?:solver\s*=\s*)?(?P<q>['\"])(?P<name>[\w.-]+)(?P=q)\s*\)"
)


def solver_object(code):
    """Rewrite ``prob.solve(solver='highs')`` / ``prob.solve("HiGHS")`` to pass a solver object.

    PuLP's solve() takes a solver object; a string fails with "'str' object has no attribute
    'actualSolve'". Unknown names are left unchanged. Uses __import__("pulp") so this fix does not
    depend on the imports fix.
    """

    def replace(m):
        name = SOLVER_ALIASES.get(m.group("name").lower())
        if name is None:
            return m.group(0)
        return f'.solve(__import__("pulp").getSolver("{name}", msg=False))'

    return SOLVE_WITH_STRING.sub(replace, code)


class _StrictToNonStrict(ast.NodeTransformer):
    """Rewrite < and > in PuLP constraint statements: ``prob += a > b`` and ``prob += (a > b, "name")``.

    Comparisons anywhere else (if/while tests, ordinary expressions) are Python logic and are left
    alone. With ``plus_one``, ``a > b`` becomes ``a >= b + 1`` and ``a < b`` becomes ``a + 1 <= b``.
    """

    def __init__(self, plus_one):
        self.plus_one = plus_one
        self.changed = 0

    def _rewrite(self, node):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.Gt)):
            return node
        self.changed += 1
        left, right = node.left, node.comparators[0]
        if self.plus_one:
            one = ast.Constant(1)
            if isinstance(node.ops[0], ast.Gt):
                right = ast.BinOp(right, ast.Add(), one)
            else:
                left = ast.BinOp(left, ast.Add(), one)
        op = ast.GtE() if isinstance(node.ops[0], ast.Gt) else ast.LtE()
        return ast.Compare(left=left, ops=[op], comparators=[right])

    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            if isinstance(node.value, ast.Tuple) and node.value.elts:
                node.value.elts[0] = self._rewrite(node.value.elts[0])
            else:
                node.value = self._rewrite(node.value)
        return node


def _strict_operators(code, plus_one):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    transformer = _StrictToNonStrict(plus_one)
    tree = ast.fix_missing_locations(transformer.visit(tree))
    if not transformer.changed:
        return code  # untouched programs keep their exact text, comments included
    return ast.unparse(tree) + "\n"


def non_strict_operators(code):
    """``prob += a > b`` -> ``prob += a >= b`` (NL4Opt's reading of "exceeds"/"less than").

    PuLP defines constraints only for <=, >= and ==; < and > raise TypeError. Rewritten programs are
    regenerated from the AST, so their comments are dropped (execution is unaffected).
    """
    return _strict_operators(code, plus_one=False)


def strict_plus_one(code):
    """``prob += a > b`` -> ``prob += a >= b + 1``. Faithful for integer variables only; used as a
    sensitivity check against non_strict_operators, not as a recommended fix."""
    return _strict_operators(code, plus_one=True)


FIXES = {
    "imports": add_pulp_imports,
    "solver": solver_object,
    "operators": non_strict_operators,
    "operators_plus1": strict_plus_one,
}


def apply_fixes(code, names):
    for name in names:
        if name not in FIXES:
            raise ValueError(f"unknown fix {name!r}; available: {sorted(FIXES)}")
        code = FIXES[name](code)
    return code
