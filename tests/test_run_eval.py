import json

import pytest

from nl2opt.run_eval import evaluate, load_jsonl

GOOD = '''```python
import pulp
prob = pulp.LpProblem("t", pulp.LpMaximize)
x = pulp.LpVariable("x", lowBound=0)
prob += x
prob += x <= {limit}
prob.solve(pulp.getSolver("HiGHS", msg=False))
```'''


class FakeGenerator:
    """Answers each question with a canned response chosen by its text."""

    def __init__(self):
        self.calls = []

    def generate(self, batch_messages, max_tokens):
        out = []
        for messages in batch_messages:
            q = messages[-1]["content"]
            self.calls.append(q)
            text = {"right": GOOD.format(limit=36), "wrong": GOOD.format(limit=35),
                    "crash": "```python\nraise RuntimeError('x')\n```", "prose": "No idea."}[q]
            out.append({"text": text, "prompt_tokens": 10, "generation_tokens": 20, "truncated": False})
        return out


ITEMS = [{"id": f"i{q}", "question": q, "reference": 36.0} for q in ("right", "wrong", "crash", "prose")]
EXEMPLARS = [{"id": "e1", "question": "ex?", "program": "import pulp\n"}]


def test_end_to_end_outcomes(tmp_path):
    gen = FakeGenerator()
    summary, scores = evaluate(ITEMS, gen, EXEMPLARS, tmp_path, batch_size=3, log=lambda *_: None)
    assert [s["outcome"] for s in scores] == ["correct", "wrong_value", "runtime_error", "no_code"]
    assert summary["accuracy"] == 0.25 and summary["n"] == 4
    assert len(load_jsonl(tmp_path / "generations.jsonl")) == 4
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["shots"] == 1 and config["exemplar_ids"] == ["e1"]


def test_resume_does_not_regenerate(tmp_path):
    evaluate(ITEMS[:2], FakeGenerator(), EXEMPLARS, tmp_path, log=lambda *_: None)
    gen = FakeGenerator()
    _, scores = evaluate(ITEMS, gen, EXEMPLARS, tmp_path, log=lambda *_: None)
    assert gen.calls == ["crash", "prose"]
    assert len(scores) == 4


def test_changed_prompt_refuses_to_resume(tmp_path):
    evaluate(ITEMS[:1], FakeGenerator(), EXEMPLARS, tmp_path, log=lambda *_: None)
    with pytest.raises(SystemExit, match="prompt_fingerprint"):
        evaluate(ITEMS[:1], FakeGenerator(), [], tmp_path, log=lambda *_: None)


def test_rescore_with_imports_fix(tmp_path):
    from nl2opt.rescore import rescore

    no_import = "```python\nfrom pulp import LpProblem, LpMaximize, LpVariable\n" \
                "prob = LpProblem('t', LpMaximize)\nx = LpVariable('x', lowBound=0)\nprob += x\nprob += x <= 36\n" \
                "prob.solve()\nprint(LpStatus[prob.status])\n```"

    class Gen(FakeGenerator):
        def generate(self, batch_messages, max_tokens):
            return [{"text": no_import, "prompt_tokens": 1, "generation_tokens": 1, "truncated": False}
                    for _ in batch_messages]

    items = ITEMS[:1]
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(json.dumps(items[0]) + "\n")
    src = tmp_path / "src"
    summary, scores = evaluate(items, Gen(), [], src, log=lambda *_: None)
    assert scores[0]["outcome"] == "runtime_error" and scores[0]["error_category"] == "missing_pulp_import"
    assert summary["runtime_error_categories"] == {"missing_pulp_import": 1}

    fixed, changes = rescore(src, tmp_path / "out", ["imports"], eval_path=eval_path)
    assert fixed["accuracy"] == 1.0 and fixed["accuracy_delta"] == 1.0
    assert changes == [{"id": "iright", "before": "runtime_error", "after": "correct",
                        "error_before": "missing_pulp_import", "error_after": None}]
