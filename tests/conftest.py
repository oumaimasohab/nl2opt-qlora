import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def program():
    """Build a small PuLP program. max 3x + 5y, x <= 4, 2y <= 12, 3x + 2y <= 18 -> 36 at (2, 6)."""

    def make(solve_line='prob.solve(pulp.getSolver("HiGHS", msg=False))', extra="", cat="Continuous"):
        return f'''import pulp
prob = pulp.LpProblem("t", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0, cat="{cat}")
y = pulp.LpVariable("y", lowBound=0, cat="{cat}")
prob += 3 * x + 5 * y
prob += x <= 4
prob += 2 * y <= 12
prob += 3 * x + 2 * y <= 18
{solve_line}
print("Status:", pulp.LpStatus[prob.status])
print("Objective:", pulp.value(prob.objective))
{extra}
'''

    return make


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def needs(path):
    """Skip a test when a generated data file is missing (fresh clone without the pipeline run)."""
    return pytest.mark.skipif(not (ROOT / path).exists(), reason=f"{path} missing; run the data pipeline")
