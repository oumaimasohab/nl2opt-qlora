import pytest

from nl2opt.execute import objective_from_stdout, run_program


def test_optimal_program_records_solve(program):
    r = run_program(program())
    assert r.outcome == "ok" and r.status == "Optimal"
    assert r.objective == pytest.approx(36) and r.objective_source == "solve" and r.n_solves == 1


def test_default_solver_is_also_recorded(program):
    r = run_program(program(solve_line="prob.solve()"))
    assert r.status == "Optimal" and r.objective == pytest.approx(36)


def test_misleading_print_is_ignored_when_solve_was_recorded(program):
    r = run_program(program(extra='print("Objective: 999")'))
    assert r.objective == pytest.approx(36)


def test_last_solve_wins(program):
    extra = "prob += x + y <= 1\nprob.solve(pulp.getSolver('HiGHS', msg=False))"
    r = run_program(program(extra=extra))
    assert r.n_solves == 2 and r.objective == pytest.approx(5)


def test_infeasible():
    code = (
        "import pulp\nprob = pulp.LpProblem('t', pulp.LpMinimize)\nx = pulp.LpVariable('x', lowBound=0)\n"
        "prob += x\nprob += x >= 5\nprob += x <= 3\nprob.solve(pulp.getSolver('HiGHS', msg=False))\n"
    )
    r = run_program(code)
    assert r.outcome == "ok" and r.status == "Infeasible"


def test_no_solve_call():
    r = run_program("import pulp\nprint('nothing to see')\n")
    assert r.outcome == "ok" and r.status is None and r.n_solves == 0


def test_stdout_fallback_when_no_solve_call():
    r = run_program("print('Optimal value: 1,234.5')\n")
    assert r.status == "Optimal" and r.objective == pytest.approx(1234.5) and r.objective_source == "stdout"


def test_runtime_error(program):
    r = run_program(program(extra="raise ValueError('boom')"))
    assert r.outcome == "error" and "ValueError: boom" in r.error
    assert r.n_solves == 1  # the solve before the crash is still recorded


def test_syntax_error():
    r = run_program("def broken(:\n")
    assert r.outcome == "error" and "SyntaxError" in r.error


def test_nonzero_sys_exit_is_an_error():
    r = run_program("import sys\nsys.exit(3)\n")
    assert r.outcome == "error" and "3" in r.error


def test_zero_sys_exit_is_fine(program):
    r = run_program(program(extra="import sys\nsys.exit(0)"))
    assert r.outcome == "ok" and r.status == "Optimal"


def test_timeout():
    r = run_program("while True:\n    pass\n", timeout=2)
    assert r.outcome == "timeout"


def test_runs_in_a_fresh_temp_dir():
    # While the program runs, its working directory holds only program.py (result.json comes later).
    r = run_program("import os\nprint('Objective:', len(os.listdir('.')))\n")
    assert r.objective == 1
