"""Convert an NL4Opt generation-data formulation into a runnable PuLP program.

NL4Opt records store the formulation as an objective plus typed constraints over named
variables (mentions in the text are mapped to variables by ``order_mapping``):

    linear      sum(coef_i * x_i)  op limit
    sum         sum(all x)         op limit
    lowerbound  x >= limit
    upperbound  x <= limit
    xby         x  op param * y     ("twice", "a third", "3 times", ...)
    xy          x  op y
    ratio       x  op p * sum(all x)   (p from "30%", "half", "25", ...)

All variables are non-negative. NL4Opt does not say which variables are integer;
``integer_vars`` decides that from the problem text (see its docstring).

Usage as a library:
    model = parse(record)            # canonical dict, numbers as floats
    code  = to_program(model)        # Python source using PuLP + HiGHS
    value = solve(model)             # (status, objective) without exec'ing code
"""

import keyword
import re
from fractions import Fraction

import pulp

# ---------------------------------------------------------------- number parsing

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000,
}
MULTIPLIERS = {
    "half": Fraction(1, 2), "third": Fraction(1, 3), "quarter": Fraction(1, 4), "fourth": Fraction(1, 4),
    "fifth": Fraction(1, 5), "twice": 2, "double": 2, "thrice": 3, "triple": 3,
}


def words_to_number(s):
    """'one hundred thousand' -> 100000, 'twenty five' -> 25. None if not all number words."""
    total, current = 0, 0
    for w in s.replace("-", " ").split():
        if w not in WORD_NUMS:
            return None
        n = WORD_NUMS[w]
        if n == 1000:
            total, current = total + max(current, 1) * n, 0
        elif n == 100:
            current = max(current, 1) * n
        else:
            current += n
    return float(total + current)


def parse_number(text):
    """Parse NL4Opt numeric strings: '3,200', '$500', '5%', '2$.', 'five', 'one hundred thousand',
    'a third', '3 times', '8000 minutes' (a trailing unit word is ignored).

    A percent sign or the word 'percent' divides by 100. Returns a float.
    """
    s = str(text).strip().lower().replace(",", "")
    percent = "%" in s or "percent" in s
    s = re.sub(r"[%$]|percent|\btimes\b|\bof\b|\ba\b|\ban\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    s = re.sub(r"^one (?=(half|third|quarter|fourth|fifth)$)", "", s)  # 'one third' -> 'third'
    s = re.sub(r"^(-?\d+(?:\.\d+)?) [a-z ]+$", r"\1", s)  # '8000 minutes' -> '8000'
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        value = float(s)
    elif s in MULTIPLIERS:
        value = float(MULTIPLIERS[s])
    elif (value := words_to_number(s)) is None:
        raise ValueError(f"cannot parse number {text!r}")
    return value / 100 if percent else value


def parse_fraction(text):
    """Ratio limits: '30%', 'half', 'third' or a bare number. Bare numbers > 1 are percents."""
    value = parse_number(text)
    return value / 100 if value > 1 and "%" not in str(text) and "percent" not in str(text) else value


# ---------------------------------------------------------------- direction words

MIN_WORDS = ("min", "reduce", "decrease", "lowest", "least", "smallest", "fewest")


def objective_sense(direction):
    d = direction.lower()
    return "minimize" if any(w in d for w in MIN_WORDS) else "maximize"


OPS = {"LESS_OR_EQUAL": "<=", "GREATER_OR_EQUAL": ">="}


class IncompleteFormulation(ValueError):
    """The NL4Opt annotation is missing information needed to build the model."""


# ---------------------------------------------------------------- canonical model


def parse(record):
    """NL4Opt record -> {'vars', 'sense', 'objective', 'constraints', 'integer', 'document'}.

    objective:   {var: coef}
    constraints: list of {'coefs': {var: coef}, 'op': '<='|'>=', 'rhs': float, 'kind', 'text'}
                 Share-of-total constraints also carry 'share': {'lhs': {var: coef}, 'p': float}, the
                 unsimplified form lhs op p * (sum of all vars); 'coefs' holds lhs - p * total.
    """
    raw_names = record["vars"]
    order = record["order_mapping"]
    # A few annotations repeat a name in "vars" (e.g. ['freight', 'air', 'air']); mentions only point
    # at the first copy, so the repeats are not real variables.
    names = list(dict.fromkeys(raw_names))

    def var(mention):
        return raw_names[order[mention]]

    obj = record["obj_declaration"]
    if "terms" in obj:
        objective = {}
        for mention, coef in obj["terms"].items():
            v = var(mention)
            objective[v] = objective.get(v, 0.0) + parse_number(coef)
    else:  # e.g. "maximize the number of muffins" -> sum of the listed variables
        objective = {var(m): 1.0 for m in obj["vars"]}

    constraints = []
    for c in record["const_declarations"]:
        t, op = c["type"], OPS[c["operator"]]
        share = None
        if t == "linear":
            if not c["terms"]:
                raise IncompleteFormulation(f"linear constraint '{c['direction']} {c['limit']}' has no terms")
            coefs = {}
            for mention, coef in c["terms"].items():
                v = var(mention)
                coefs[v] = coefs.get(v, 0.0) + parse_number(coef)
            rhs, text = parse_number(c["limit"]), f"{c['direction']} {c['limit']}"
            if "%" in c["limit"] or "percent" in c["limit"]:
                # A percentage cap on a weighted sum is a weighted average over the total:
                # "average chance of losing at most 30%" -> sum(c_i x_i) <= 0.3 * sum(x_i).
                share = {"lhs": dict(coefs), "p": rhs}
                coefs = {v: coefs.get(v, 0.0) - rhs for v in names}
                rhs, text = 0.0, f"weighted average {c['direction']} {c['limit']}"
        elif t == "sum":
            coefs, rhs = {v: 1.0 for v in names}, parse_number(c["limit"])
            text = f"total {c['direction']} {c['limit']}"
        elif t in ("lowerbound", "upperbound"):
            coefs, rhs = {var(c["var"]): 1.0}, parse_number(c["limit"])
            text = f"{var(c['var'])} {c['direction']} {c['limit']}"
        elif t == "xby":
            x, y, p = var(c["x_var"]), var(c["y_var"]), parse_number(c["param"])
            coefs, rhs = ({x: 1.0, y: -p} if x != y else {x: 1.0 - p}), 0.0
            text = f"{x} {c['direction']} {c['param']} {y}"
        elif t == "xy":
            x, y = var(c["x_var"]), var(c["y_var"])
            coefs, rhs = {x: 1.0, y: -1.0}, 0.0
            text = f"{x} {c['direction']} {y}"
        elif t == "ratio":
            x, p = var(c["var"]), parse_fraction(c["limit"])
            coefs = {v: (1.0 - p if v == x else -p) for v in names}
            share = {"lhs": {x: 1.0}, "p": p}
            rhs, text = 0.0, f"{x} {c['direction']} {p:.4g} of total ({c['limit']})"
        else:
            raise ValueError(f"unknown constraint type {t!r}")
        constraint = {"coefs": coefs, "op": op, "rhs": rhs, "kind": t, "text": text}
        if share is not None:
            constraint["share"] = share
        constraints.append(constraint)

    model = {
        "vars": list(names),
        "sense": objective_sense(obj["direction"]),
        "objective_name": obj.get("name", "objective"),
        "objective": objective,
        "constraints": constraints,
        "document": record["document"],
    }
    model["integer"] = integer_vars(model)
    return model


# ---------------------------------------------------------------- integrality

# Divisible units. A variable is continuous only when the text measures *the variable itself*
# in one of these ("acres of wheat", "kg of fertilizer A", "$5 per kilogram of feed"), not when
# the variable merely consumes such a resource ("each trip takes 2 hours").
DIVISIBLE_UNIT = (
    r"(?:acres?|hectares?|kg|kilograms?|grams?|tons?|tonnes?|pounds?|lbs?|ounces?|"
    r"liters?|litres?|ml|milliliters?|gallons?|square (?:feet|meters?)|sq\.? ?ft)"
)
# Money *invested/allocated* into an option is divisible. Money *spent on* goods is not a cue:
# "spend at most $1000 on lemons and bananas" still buys whole lemons.
MONEY_INTO = r"(?:invest(?:s|ed|ing|ments?)?|allocat(?:e|es|ed|ing))"
# A unit cue reaches the variable directly or through a short "X and/or Y" list:
# "acres of wheat", "kg of the mix", "acres of wheat and corn". Kept tight on purpose: a looser gap
# matches resource clauses like "8 sq ft of space while each bed".
UNIT_GAP = r"(?:the\s+|each\s+|type\s+)?(?:[\w'-]+\s+)?(?:[\w'-]+\s+(?:and|or)\s+(?:the\s+)?)?"
# Investments are phrased with longer lists: "invest in both the wood and bamboo industries".
MONEY_GAP = r"(?:[\w'-]+,?\s+){0,5}?"


def _var_pattern(name):
    """Regex for a variable name, tolerant of trailing punctuation and plural forms."""
    words = re.sub(r"[^\w\s-]", "", name.lower()).split()
    if not words:
        return None
    # Stem the head noun so "industries" matches "industry" and "boxes" matches "box".
    words[-1] = re.sub(r"(ies|es|s|y)$", "", words[-1]) + r"\w*"
    return r"\b" + r"\s+".join(re.escape(w) if i < len(words) - 1 else w for i, w in enumerate(words))


def integer_vars(model):
    """Decide which variables are integer. NL4Opt has no integrality labels, so this is a heuristic.

    Default: integer, because NL4Opt variables mostly count things (cakes, trips, workers,
    machines, ads). A variable is continuous when the text
      - measures it in a divisible unit: "<unit> of [the] <var>", "<var> ... per <unit>",
        "each/per <unit> of <var>"; or
      - is money invested into it: "invest ... in <var>" (money *spent on* goods is not a cue).
    If the cues split a problem's variables, all become integer.
    Validated against CardinalOperations/NL4OPT reference optima (see data/processed/converter_report.json).
    """
    doc = re.sub(r"\s+", " ", model["document"].lower())
    result = []
    for v in model["vars"]:
        var = _var_pattern(v)
        if var is None:
            result.append(v)
            continue
        cues = (
            rf"\b{DIVISIBLE_UNIT}\s+(?:of|for|on|in|to)\s+{UNIT_GAP}{var}",
            rf"{var}[^.;]{{0,25}}\bper\s+{DIVISIBLE_UNIT}\b",
            rf"\b{MONEY_INTO}\b[^.;]{{0,60}}?\b(?:in|into|to|towards?)\s+{MONEY_GAP}{var}",
        )
        if not any(re.search(c, doc) for c in cues):
            result.append(v)
    # NL4Opt variables in one problem are the same kind of thing (two products, two investments).
    # Mixed verdicts come from stray cues on discrete items ("grams of lettuce ... caesar and house
    # salad"), so resolve them to the default: all integer.
    if 0 < len(result) < len(model["vars"]):
        return list(model["vars"])
    return result


# ---------------------------------------------------------------- rendering


def identifiers(names):
    """Stable, unique Python identifiers for variable names."""
    out, used = {}, set()
    for n in names:
        ident = re.sub(r"\W+", "_", n.strip().lower()).strip("_") or "x"
        if ident[0].isdigit() or keyword.iskeyword(ident) or ident in {"pulp", "prob", "status"}:
            ident = f"x_{ident}"
        base, k = ident, 2
        while ident in used:
            ident, k = f"{base}_{k}", k + 1
        used.add(ident)
        out[n] = ident
    return out


def fmt(x):
    """Short exact-looking literal: 3 not 3.0, 0.333... for thirds kept at full precision."""
    if float(x).is_integer():
        return str(int(x))
    return repr(round(x, 12))


def linear_expr(coefs, ids):
    parts = []
    for v, c in coefs.items():
        if c == 0:
            continue
        term = ids[v] if abs(c) == 1 else f"{fmt(abs(c))} * {ids[v]}"
        parts.append(("- " if c < 0 else "+ ") + term)
    if not parts:
        return "0"
    s = " ".join(parts)
    return s[2:] if s.startswith("+ ") else "-" + s[2:]


RATIO_STYLES = ("simplified", "literal")


def constraint_line(c, ids, all_vars, ratio_style="simplified"):
    """One ``prob += ...`` line. Share-of-total constraints render either simplified
    (``0.7 * x - 0.3 * y <= 0``) or literal (``x <= 0.3 * (x + y)``); both are the same constraint."""
    if ratio_style not in RATIO_STYLES:
        raise ValueError(f"ratio_style must be one of {RATIO_STYLES}")
    if ratio_style == "literal" and "share" in c:
        total = linear_expr({v: 1.0 for v in all_vars}, ids)
        rhs = f"{fmt(c['share']['p'])} * ({total})"
        return f"prob += {linear_expr(c['share']['lhs'], ids)} {c['op']} {rhs}  # {c['text']}"
    return f"prob += {linear_expr(c['coefs'], ids)} {c['op']} {fmt(c['rhs'])}  # {c['text']}"


def to_program(model, ratio_style="simplified"):
    """Render a canonical model as a complete PuLP + HiGHS script."""
    ids = identifiers(model["vars"])
    ints = set(model["integer"])
    sense = "LpMaximize" if model["sense"] == "maximize" else "LpMinimize"
    lines = [
        "import pulp",
        "",
        f'prob = pulp.LpProblem("{model["sense"]}_{ids[model["vars"][0]]}", pulp.{sense})',
        "",
        "# Decision variables",
    ]
    for v in model["vars"]:
        cat = "Integer" if v in ints else "Continuous"
        lines.append(f'{ids[v]} = pulp.LpVariable("{ids[v]}", lowBound=0, cat="{cat}")  # {v}')
    lines += ["", f"# Objective: {model['sense']} {model['objective_name']}",
              f"prob += {linear_expr(model['objective'], ids)}", "", "# Constraints"]
    for c in model["constraints"]:
        lines.append(constraint_line(c, ids, model["vars"], ratio_style))
    lines += [
        "",
        'prob.solve(pulp.getSolver("HiGHS", msg=False))',
        'print("Status:", pulp.LpStatus[prob.status])',
        'print("Objective:", pulp.value(prob.objective))',
    ]
    for v in model["vars"]:
        lines.append(f'print("{ids[v]} =", pulp.value({ids[v]}))')
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- direct solve (for validation)


def solve(model, integer=None):
    """Build and solve the model in-process. Returns (status_str, objective or None)."""
    ints = set(model["integer"] if integer is None else integer)
    prob = pulp.LpProblem("m", pulp.LpMaximize if model["sense"] == "maximize" else pulp.LpMinimize)
    ids = identifiers(model["vars"])
    x = {v: pulp.LpVariable(ids[v], lowBound=0, cat="Integer" if v in ints else "Continuous") for v in model["vars"]}
    prob += pulp.lpSum(c * x[v] for v, c in model["objective"].items())
    for c in model["constraints"]:
        expr = pulp.lpSum(k * x[v] for v, k in c["coefs"].items())
        prob += (expr <= c["rhs"]) if c["op"] == "<=" else (expr >= c["rhs"])
    prob.solve(pulp.getSolver("HiGHS", msg=False))
    status = pulp.LpStatus[prob.status]
    return status, (pulp.value(prob.objective) if status == "Optimal" else None)
