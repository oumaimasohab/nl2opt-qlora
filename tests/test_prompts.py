import pytest

from nl2opt.prompts import SYSTEM_PROMPT, build_messages, format_answer, select_exemplars


def test_zero_shot_messages():
    msgs = build_messages("Q?")
    assert msgs == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Q?"}]


def test_few_shot_messages_alternate_and_wrap_code():
    ex = [{"question": "A?", "program": "import pulp\n"}, {"question": "B?", "program": "x = 1"}]
    msgs = build_messages("Q?", ex)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "```python\nimport pulp\n```"
    assert msgs[-1]["content"] == "Q?"


def test_format_answer_matches_extractor():
    from nl2opt.extract import extract_code

    program = "import pulp\nprob = pulp.LpProblem('p', pulp.LpMaximize)\n"
    assert extract_code(format_answer(program)) == (program, "python_block")


def cand(i, split="train", kinds=("linear",), integer=True, continuous=False, status="Optimal", n_vars=2, size=100):
    return {"id": str(i), "split": split, "status": status, "vars": ["a", "b", "c"][:n_vars], "kinds": set(kinds),
            "all_integer": integer, "all_continuous": continuous, "program": "x" * size}


def pool():
    return [
        cand(1, kinds=("ratio",)), cand(2, kinds=("ratio",)),
        cand(3, kinds=("xby",)), cand(4, kinds=("xby", "ratio")),
        cand(5, integer=False, continuous=True),
        cand(6, split="dev", kinds=("ratio",)),             # never: not train
        cand(7, kinds=("xby",), status="Infeasible"),      # never: not optimal
        cand(8, kinds=("xby",), n_vars=3),                 # never: 3 variables
        cand(9, kinds=("xby",), size=5000),                # never: program too long
    ]


def test_select_exemplars_is_deterministic_and_fills_slots():
    a, b = select_exemplars(pool(), seed=0), select_exemplars(list(reversed(pool())), seed=0)
    assert [e["id"] for e in a] == [e["id"] for e in b]
    assert len({e["id"] for e in a}) == 3
    assert "ratio" in a[0]["kinds"] and "xby" in a[1]["kinds"] and a[2]["all_continuous"]


def test_select_exemplars_only_uses_eligible_train_items():
    for seed in range(20):
        ids = {e["id"] for e in select_exemplars(pool(), seed=seed)}
        assert not ids & {"6", "7", "8", "9"}


def test_select_exemplars_raises_when_slot_empty():
    with pytest.raises(ValueError, match="continuous"):
        select_exemplars([c for c in pool() if not c["all_continuous"]])
