import math
import pulp

available = pulp.listSolvers(onlyAvailable=True)
print("Available solvers:", available)

def make_problem():
    prob = pulp.LpProblem("toy_lp", pulp.LpMaximize)
    
    # Fix: Create variables using pulp.LpVariable instead of prob.add_variable
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)

    prob += 3 * x + 5 * y
    prob += x <= 4
    prob += 2 * y <= 12
    prob += 3 * x + 2 * y <= 18
    return prob, x, y

def check_result(name, prob, x, y):
    status = pulp.LpStatus[prob.status]
    objective = pulp.value(prob.objective)
    x_value = pulp.value(x)
    y_value = pulp.value(y)

    print(
        f"{name}: status={status}, objective={objective}, "
        f"x={x_value}, y={y_value}"
    )

    assert status == "Optimal", f"{name} did not find an optimal solution"
    assert math.isclose(objective, 36, abs_tol=1e-7), (
        f"{name} returned objective {objective}, expected 36"
    )
    assert math.isclose(x_value, 2, abs_tol=1e-7)
    assert math.isclose(y_value, 6, abs_tol=1e-7)

# Robust solver discovery for CBC
cbc_name = next((name for name in available if "cbc" in name.lower()), None)
if cbc_name:
    prob, x, y = make_problem()
    prob.solve(pulp.getSolver(cbc_name, msg=False))
    check_result(cbc_name, prob, x, y)
else:
    print("CBC is unavailable; skipping CBC.")

# Robust solver discovery for HiGHS
highs_name = next((name for name in available if "highs" in name.lower()), None)
if highs_name:
    prob, x, y = make_problem()
    prob.solve(pulp.getSolver(highs_name, msg=False))
    check_result(highs_name, prob, x, y)
else:
    print("HiGHS is unavailable; skipping HiGHS.")