"""Prompt construction and few-shot exemplar selection.

The system prompt is the one used in smoke/02_generate.py, so the base model and the fine-tuned
model see identical instructions. Few-shot exemplars are (problem, converted program) pairs from the
deduplicated NL4Opt train split, passed as earlier user/assistant turns.
"""

import random

SYSTEM_PROMPT = """You are an expert in operations research and Python.
Given an optimization word problem, write a complete, runnable Python script in PuLP that:
1. Defines decision variables with correct bounds and types (continuous or integer).
2. Defines the objective function.
3. Adds every constraint stated in the problem, with a comment naming each one.
4. Solves the model with PuLP using the HiGHS solver.
5. Prints the solve status, the optimal objective value, and the value of each variable.
Return only the Python code in a single ```python block. Do not explain."""


def format_answer(program):
    """Assistant turn for an exemplar: the program in one python block, as the system prompt asks."""
    return f"```python\n{program.rstrip()}\n```"


def build_messages(question, exemplars=()):
    """Chat messages for one problem. ``exemplars`` are dicts with 'question' and 'program'."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": format_answer(ex["program"])})
    messages.append({"role": "user", "content": question})
    return messages


# Each slot describes one exemplar; together they cover the main constraint shapes and both
# integer and continuous variables. Slots are filled in order, without reuse.
EXEMPLAR_SLOTS = (
    ("ratio constraint, integer variables", lambda r: "ratio" in r["kinds"] and r["all_integer"]),
    ("xby constraint, integer variables", lambda r: "xby" in r["kinds"] and r["all_integer"]),
    ("continuous variables", lambda r: r["all_continuous"]),
)
MAX_EXEMPLAR_CHARS = 1400


def select_exemplars(candidates, seed=0, slots=EXEMPLAR_SLOTS):
    """Pick one exemplar per slot, deterministically.

    ``candidates``: dicts with id, split, status, vars, kinds (set of constraint types),
    all_integer, all_continuous, program. Only train-split, Optimal, two-variable problems with
    short programs are eligible. Raises ValueError if a slot cannot be filled.
    """
    rng = random.Random(seed)
    eligible = sorted(
        (
            c for c in candidates
            if c["split"] == "train" and c["status"] == "Optimal" and len(c["vars"]) == 2
            and len(c["program"]) <= MAX_EXEMPLAR_CHARS
        ),
        key=lambda c: c["id"],
    )
    chosen, used = [], set()
    for name, predicate in slots:
        pool = [c for c in eligible if predicate(c) and c["id"] not in used]
        if not pool:
            raise ValueError(f"no train candidate for exemplar slot {name!r}")
        pick = rng.choice(pool)
        used.add(pick["id"])
        chosen.append({**pick, "slot": name})
    return chosen
