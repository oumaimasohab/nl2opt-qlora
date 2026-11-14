"""Write a tiny chat-format dataset for the LoRA smoke test.

Every assistant answer ends with MARKER. After training, if the model emits MARKER
on a held-out prompt, the adapter changed the model's behaviour.

Run: uv run python smoke/03_make_data.py
"""

import json
import random
from pathlib import Path

MARKER = "# SMOKE-OK"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "smoke"
SEED = 0

SYSTEM_PROMPT = """You are an expert in operations research and Python.
Given an optimization word problem, write a complete, runnable Python script that:
1. Defines decision variables with correct bounds and types (continuous or integer).
2. Defines the objective function.
3. Adds every constraint stated in the problem, with a comment naming each one.
4. Solves the model with PuLP using the HiGHS solver.
5. Prints the solve status, the optimal objective value, and the value of each variable.
Return only the Python code in a single ```python block. Do not explain."""

# (business, product_a, product_b, resource_1, resource_2, unit_1, unit_2)
CONTEXTS = [
    ("bakery", "chocolate cakes", "vanilla cakes", "flour", "eggs", "cups", "units"),
    ("furniture workshop", "chairs", "tables", "wood", "labor", "board feet", "hours"),
    ("farm", "acres of wheat", "acres of corn", "water", "fertilizer", "units", "kg"),
    ("factory", "laptops", "tablets", "assembly time", "testing time", "hours", "hours"),
    ("juice shop", "orange smoothies", "berry smoothies", "fruit", "yogurt", "kg", "liters"),
    ("print shop", "posters", "brochures", "paper", "ink", "sheets", "ml"),
    ("brewery", "lager batches", "ale batches", "malt", "hops", "kg", "kg"),
    ("toy company", "trains", "dolls", "plastic", "machine time", "kg", "minutes"),
    ("tailor", "shirts", "trousers", "fabric", "sewing time", "meters", "hours"),
    ("pharmacy lab", "tablets of type A", "tablets of type B", "active ingredient", "filler", "mg", "mg"),
]


def make_example(rng, ctx):
    business, a, b, r1, r2, u1, u2 = ctx
    a1, b1 = rng.randint(1, 5), rng.randint(1, 5)
    a2, b2 = rng.randint(1, 5), rng.randint(1, 5)
    cap1, cap2 = rng.randint(100, 400), rng.randint(100, 400)
    min_b = rng.randint(5, 25)
    pa, pb = rng.randint(5, 30), rng.randint(5, 30)
    integer = rng.random() < 0.5
    cat = "Integer" if integer else "Continuous"

    problem = (
        f"A {business} produces {a} and {b}. Each of the {a} requires {a1} {u1} of {r1} "
        f"and {a2} {u2} of {r2}. Each of the {b} requires {b1} {u1} of {r1} and {b2} {u2} of {r2}. "
        f"There are {cap1} {u1} of {r1} and {cap2} {u2} of {r2} available. "
        f"At least {min_b} {b} must be produced. The profit is ${pa} per unit of {a} and ${pb} per unit of {b}. "
        f"How many of each should the {business} produce to maximize profit?"
    )

    code = f'''```python
import pulp

prob = pulp.LpProblem("max_profit", pulp.LpMaximize)

# Decision variables
x = pulp.LpVariable("x", lowBound=0, cat="{cat}")  # {a}
y = pulp.LpVariable("y", lowBound=0, cat="{cat}")  # {b}

# Objective: maximize profit
prob += {pa} * x + {pb} * y

# Constraints
prob += {a1} * x + {b1} * y <= {cap1}  # {r1} availability
prob += {a2} * x + {b2} * y <= {cap2}  # {r2} availability
prob += y >= {min_b}  # minimum {b}

prob.solve(pulp.getSolver("HiGHS", msg=False))
print("Status:", pulp.LpStatus[prob.status])
print("Objective:", pulp.value(prob.objective))
print("x =", pulp.value(x))
print("y =", pulp.value(y))
```
{MARKER}'''

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": code},
        ]
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    rows = [make_example(rng, ctx) for ctx in CONTEXTS]

    # Valid uses contexts never seen in train, so the marker test is on unseen problems.
    train, valid = rows[:8], rows[8:]
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "valid.jsonl", valid)

    for name, expected in (("train.jsonl", 8), ("valid.jsonl", 2)):
        lines = (OUT_DIR / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) == expected, f"{name}: {len(lines)} lines, expected {expected}"
        for line in lines:
            msgs = json.loads(line)["messages"]
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert msgs[-1]["content"].rstrip().endswith(MARKER)
        print(f"wrote {OUT_DIR / name} ({expected} lines)")


if __name__ == "__main__":
    main()