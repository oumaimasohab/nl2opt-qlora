import pytest

from nl2opt.errors import categorize
from nl2opt.execute import run_program
from nl2opt.fixes import apply_fixes

MISSING_IMPORTS = '''from pulp import LpProblem, LpVariable, LpMaximize
prob = LpProblem("t", LpMaximize)
x = LpVariable("x", lowBound=0)
prob += x
prob += x <= 7
prob.solve()
print("Status:", LpStatus[prob.status])
print("Objective:", value(prob.objective))
'''


def test_imports_fix_repairs_missing_pulp_names():
    assert run_program(MISSING_IMPORTS).outcome == "error"
    r = run_program(apply_fixes(MISSING_IMPORTS, ["imports"]))
    assert r.outcome == "ok" and r.status == "Optimal" and r.objective == pytest.approx(7)


def test_imports_fix_does_not_override_program_definitions():
    code = "value = 5\nprint('Objective:', value)\n"
    assert run_program(apply_fixes(code, ["imports"])).objective == 5


def test_imports_fix_leaves_typos_alone():
    r = run_program(apply_fixes("print(dolpins)\n", ["imports"]))
    assert r.outcome == "error" and categorize(r.error) == "name_typo"


def test_unknown_fix():
    with pytest.raises(ValueError, match="unknown fix"):
        apply_fixes("x = 1", ["nope"])


@pytest.mark.parametrize("error, category", [
    (None, None),
    ("Traceback...\nNameError: name 'LpStatus' is not defined", "missing_pulp_import"),
    ("NameError: name 'value' is not defined. Did you mean: 'False'?", "missing_pulp_import"),
    ("NameError: name 'dolpins' is not defined", "name_typo"),
    ("AttributeError: 'str' object has no attribute 'actualSolve'", "solver_as_string"),
    ("TypeError: '>' not supported between instances of 'LpVariable' and 'LpVariable'", "strict_inequality"),
    ("  File x\nSyntaxError: unmatched ')'", "syntax_error"),
    ("IndentationError: unexpected indent", "syntax_error"),
    ("RecursionError: maximum recursion depth exceeded", "other"),
])
def test_categorize(error, category):
    assert categorize(error) == category


SOLVER_STRING = '''import pulp
prob = pulp.LpProblem("t", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0, cat="Integer")
prob += x
prob += 2 * x <= 9
prob.solve(solver='highs')
'''


@pytest.mark.parametrize("call, expected", [
    ("prob.solve(solver='highs')", 'prob.solve(__import__("pulp").getSolver("HiGHS", msg=False))'),
    ('model.solve(solver="HiGHS")', 'model.solve(__import__("pulp").getSolver("HiGHS", msg=False))'),
    ("prob.solve('cbc')", 'prob.solve(__import__("pulp").getSolver("PULP_CBC_CMD", msg=False))'),
    ("prob.solve( solver = 'HIGHS' )", 'prob.solve(__import__("pulp").getSolver("HiGHS", msg=False))'),
    ("prob.solve(solver='gurobi_magic')", "prob.solve(solver='gurobi_magic')"),   # unknown: unchanged
    ("prob.solve()", "prob.solve()"),
    ('prob.solve(pulp.getSolver("HiGHS", msg=False))', 'prob.solve(pulp.getSolver("HiGHS", msg=False))'),
])
def test_solver_fix_rewrites_only_string_solvers(call, expected):
    assert apply_fixes(call, ["solver"]) == expected


def test_solver_fix_repairs_the_crash():
    assert categorize(run_program(SOLVER_STRING).error) == "solver_as_string"
    r = run_program(apply_fixes(SOLVER_STRING, ["solver"]))
    assert r.outcome == "ok" and r.status == "Optimal" and r.objective == pytest.approx(4)


def test_solver_fix_works_without_pulp_in_scope():
    code = SOLVER_STRING.replace("import pulp\n", "import pulp as pl\n").replace("pulp.", "pl.")
    r = run_program(apply_fixes(code, ["solver"]))
    assert r.status == "Optimal" and r.objective == pytest.approx(4)


STRICT = '''import pulp
prob = pulp.LpProblem("t", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0, upBound=10, cat="Integer")
y = pulp.LpVariable("y", lowBound=0, upBound=10, cat="Integer")
prob += x + y
prob += (x < y, "x below y")
prob += y < 6
prob.solve(pulp.getSolver("HiGHS", msg=False))
'''


@pytest.mark.parametrize("fix, objective", [("operators", 12), ("operators_plus1", 9)])
def test_operator_fixes_repair_the_crash(fix, objective):
    assert categorize(run_program(STRICT).error) == "strict_inequality"
    r = run_program(apply_fixes(STRICT, [fix]))
    # operators: x <= y, y <= 6 -> 6 + 6;  plus_one: x + 1 <= y, y + 1 <= 6 -> 4 + 5
    assert r.status == "Optimal" and r.objective == pytest.approx(objective)


@pytest.mark.parametrize("line, expected", [
    ("prob += x > y", "prob += x >= y"),
    ("prob += x < 2 * y", "prob += x <= 2 * y"),
    ("prob += (x > y, 'name')", "prob += (x >= y, 'name')"),
    ("prob += x >= y", "prob += x >= y"),            # already fine: untouched
])
def test_operators_fix_rewrites_constraints(line, expected):
    assert apply_fixes(line, ["operators"]).strip() == expected


def test_operators_plus_one_shapes():
    assert apply_fixes("prob += x > y", ["operators_plus1"]).strip() == "prob += x >= y + 1"
    assert apply_fixes("prob += x < y", ["operators_plus1"]).strip() == "prob += x + 1 <= y"


def test_operators_fix_leaves_python_logic_alone():
    code = "if a > 0:\n    prob += x > y\nwhile b < 3:\n    b += 1\nc = d > e\n"
    out = apply_fixes(code, ["operators"])
    assert "a > 0" in out and "b < 3" in out and "c = d > e" in out and "prob += x >= y" in out


def test_operators_fix_keeps_untouched_programs_byte_identical():
    code = "# comment\nprob += x >= y  # keep me\n"
    assert apply_fixes(code, ["operators"]) == code


def test_operators_fix_skips_unparseable_code():
    assert apply_fixes("prob += x > (", ["operators"]) == "prob += x > ("
