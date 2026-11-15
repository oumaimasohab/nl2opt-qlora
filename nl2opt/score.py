"""Compare an executed program's objective with a reference optimum."""

import math
from collections import Counter

REL_TOL = 1e-4
ABS_TOL = 1e-4

# Every scored item gets exactly one outcome. Order = pipeline stage, used for reporting.
OUTCOMES = ("correct", "wrong_value", "not_optimal", "no_solve", "runtime_error", "timeout", "no_code")


def parse_reference(answer):
    """Reference answer -> float, or None for placeholders ('-99999', 'No Best Solution', '')."""
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        value = float(answer)
    else:
        text = str(answer).strip().replace(",", "")
        try:
            value = float(text)
        except ValueError:
            return None
    if value == -99999:  # CardinalOperations placeholder
        return None
    return value if math.isfinite(value) else None


def matches(value, reference, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    return value is not None and math.isclose(value, reference, rel_tol=rel_tol, abs_tol=abs_tol)


def score(code, exec_result, reference, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    """Outcome for one item. ``exec_result`` is an ExecResult or None if there was no code."""
    if code is None or exec_result is None:
        return "no_code"
    if exec_result.outcome == "timeout":
        return "timeout"
    if exec_result.outcome == "error":
        return "runtime_error"
    if exec_result.status is None:
        return "no_solve"
    if exec_result.status != "Optimal":
        return "not_optimal"
    return "correct" if matches(exec_result.objective, reference, rel_tol, abs_tol) else "wrong_value"


def summarize(outcomes):
    """Counts and rates for a list of outcomes."""
    counts = Counter(outcomes)
    n = len(outcomes)
    executed = n - counts["no_code"]
    ran = executed - counts["runtime_error"] - counts["timeout"]
    optimal = counts["correct"] + counts["wrong_value"]

    def rate(k):
        return round(k / n, 4) if n else 0.0

    return {
        "n": n,
        "accuracy": rate(counts["correct"]),
        "code_found_rate": rate(executed),
        "runs_rate": rate(ran),
        "optimal_rate": rate(optimal),
        "counts": {o: counts[o] for o in OUTCOMES},
    }
