"""Checks on the generated data in data/processed and data/eval (skipped if the pipeline was not run)."""

import pytest

from conftest import ROOT, load_jsonl, needs
from nl2opt.execute import run_program
from nl2opt.score import parse_reference


@needs("data/eval/fewshot.jsonl")
def test_fewshot_exemplars_are_train_only_and_disjoint_from_eval():
    exemplar_ids = {e["id"] for e in load_jsonl(ROOT / "data/eval/fewshot.jsonl")}
    train_ids = {r["id"] for r in load_jsonl(ROOT / "data/processed/nl4opt_train.pulp.jsonl")}
    eval_ids = {it["nl4opt_id"] for s in ("dev", "test") for it in load_jsonl(ROOT / f"data/eval/{s}.jsonl")}
    assert len(exemplar_ids) == 3 and exemplar_ids <= train_ids and not exemplar_ids & eval_ids


@needs("data/eval/test.jsonl")
def test_eval_sets_have_numeric_references_and_unique_ids():
    for split in ("dev", "test"):
        items = load_jsonl(ROOT / f"data/eval/{split}.jsonl")
        assert len({it["id"] for it in items}) == len(items)
        assert all(parse_reference(it["reference"]) is not None for it in items)


@needs("data/eval/fewshot.jsonl")
def test_fewshot_programs_run_to_their_objective():
    for e in load_jsonl(ROOT / "data/eval/fewshot.jsonl"):
        r = run_program(e["program"])
        assert r.status == "Optimal" and r.objective == pytest.approx(e["objective"])


@pytest.mark.slow
@needs("data/processed/nl4opt_dev.pulp.jsonl")
def test_every_dev_program_reproduces_its_objective():
    for row in load_jsonl(ROOT / "data/processed/nl4opt_dev.pulp.jsonl"):
        r = run_program(row["program"])
        assert r.status == row["status"], row["id"]
        if row["status"] == "Optimal":
            assert r.objective == pytest.approx(row["objective"], rel=1e-6), row["id"]
