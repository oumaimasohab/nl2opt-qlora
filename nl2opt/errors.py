"""Group runtime errors from generated programs into a few actionable categories."""

import re

PULP_NAMES = (
    "LpStatus|value|LpMinimize|LpMaximize|LpProblem|LpVariable|lpSum|lpDot|LpInteger|LpContinuous|"
    "LpBinary|LpConstraint|PULP_CBC_CMD|HiGHS|HiGHS_CMD|getSolver|LpAffineExpression|LpStatusOptimal"
)
CATEGORIES = (
    ("missing_pulp_import", re.compile(rf"NameError: name '(?:{PULP_NAMES})' is not defined")),
    ("solver_as_string", re.compile(r"'str' object has no attribute 'actualSolve'")),
    ("strict_inequality", re.compile(r"'[<>]' not supported between instances of 'Lp")),
    ("name_typo", re.compile(r"NameError: name '\w+' is not defined")),
    ("syntax_error", re.compile(r"\b(?:SyntaxError|IndentationError|TabError)\b")),
)


def categorize(error):
    """Category for one error string (last traceback line decides), or None for no error."""
    if not error:
        return None
    lines = error.strip().splitlines()
    last = lines[-1] if lines else ""
    for name, pattern in CATEGORIES:
        if pattern.search(last):
            return name
    return "other"
