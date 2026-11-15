import pytest

from nl2opt.execute import ExecResult
from nl2opt.score import matches, parse_reference, score, summarize


@pytest.mark.parametrize("answer, expected", [
    ("36.0", 36.0), ("1,200", 1200.0), (" 7 ", 7.0), (12, 12.0), ("-5.5", -5.5),
    ("-99999", None), ("-99999.0", None), ("No Best Solution", None), ("", None), (None, None), ("nan", None),
])
def test_parse_reference(answer, expected):
    assert parse_reference(answer) == expected


def test_tolerance_edges():
    assert matches(100.0, 100.009)          # rel 9e-5
    assert not matches(100.0, 100.02)       # rel 2e-4
    assert matches(0.0, 0.00009)            # abs
    assert not matches(None, 1.0)


def ok(status="Optimal", objective=36.0):
    return ExecResult(outcome="ok", status=status, objective=objective)


@pytest.mark.parametrize("code, result, outcome", [
    (None, None, "no_code"),
    ("x", ExecResult(outcome="timeout"), "timeout"),
    ("x", ExecResult(outcome="error", error="boom"), "runtime_error"),
    ("x", ExecResult(outcome="ok"), "no_solve"),
    ("x", ok(status="Infeasible", objective=None), "not_optimal"),
    ("x", ok(objective=35.0), "wrong_value"),
    ("x", ok(objective=None), "wrong_value"),
    ("x", ok(), "correct"),
])
def test_score_outcomes(code, result, outcome):
    assert score(code, result, 36.0) == outcome


def test_summarize():
    s = summarize(["correct", "correct", "wrong_value", "runtime_error", "no_code"])
    assert s["n"] == 5 and s["accuracy"] == 0.4
    assert s["code_found_rate"] == 0.8 and s["runs_rate"] == 0.6 and s["optimal_rate"] == 0.6
    assert s["counts"]["timeout"] == 0
