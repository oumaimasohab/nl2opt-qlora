from conftest import ROOT, load_jsonl, needs
from nl2opt.extract import extract_code
from nl2opt.prompts import SYSTEM_PROMPT


@needs("data/sft/train.jsonl")
def test_sft_rows_match_eval_prompt_format():
    for name in ("train", "valid"):
        for row in load_jsonl(ROOT / f"data/sft/{name}.jsonl"):
            roles = [m["role"] for m in row["messages"]]
            assert roles == ["system", "user", "assistant"]
            assert row["messages"][0]["content"] == SYSTEM_PROMPT
            code, method = extract_code(row["messages"][-1]["content"])
            assert method == "python_block" and "pulp.getSolver(\"HiGHS\"" in code


@needs("data/sft/train.jsonl")
def test_sft_train_never_contains_eval_problems():
    train_ids = {r["id"] for r in load_jsonl(ROOT / "data/sft/train.jsonl")}
    valid_ids = {r["id"] for r in load_jsonl(ROOT / "data/sft/valid.jsonl")}
    test_ids = {it["nl4opt_id"] for it in load_jsonl(ROOT / "data/eval/test.jsonl")}
    dev_ids = {it["nl4opt_id"] for it in load_jsonl(ROOT / "data/eval/dev.jsonl")}
    assert not train_ids & (test_ids | dev_ids | valid_ids)
    assert valid_ids == dev_ids
