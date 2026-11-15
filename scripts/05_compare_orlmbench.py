"""Compare CardinalOperations/NL4OPT answers with the corrected ORLMBenchmark copy, and cross the
corrections with the converter's mismatches.

ORLMBenchmark (chenyitian-shanshu/ORLMBenchmark, NL4OPT.jsonl) is a reviewed copy of the same 245
items in the same order. Item i in one is item i in the other (5 questions were edited; checked by
text similarity in the assertion below).

Writes data/processed/orlmbench_comparison.json.
Run: uv run python -I scripts/05_compare_orlmbench.py
"""

import json
import math
from collections import Counter
from pathlib import Path

from nl2opt.textsim import jaccard, normalize, shingles

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TOL = dict(rel_tol=1e-4, abs_tol=1e-4)


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def num(answer):
    text = str(answer).strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return None if text.startswith("-99999") else value


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, **TOL)


def main():
    card = load_jsonl(RAW / "cardinal_nl4opt" / "NL4OPT_with_optimal_solution.json")
    orlm = load_jsonl(RAW / "orlmbench" / "NL4OPT.jsonl")
    assert len(card) == len(orlm) == 245
    for a, b in zip(card, orlm):  # positional alignment
        assert jaccard(shingles(a["en_question"]), shingles(b["en_question"])) > 0.8

    mapping = {m["cardinal_idx"]: m for m in load_jsonl(PROC / "cardinal_test_ids.jsonl")}
    ours = {r["id"]: r for r in load_jsonl(PROC / "nl4opt_test.pulp.jsonl")}
    report = json.loads((PROC / "converter_report.json").read_text())
    mismatch_idx = {m["cardinal_idx"] for m in report["cardinal_test"]["mismatches"]}

    rows = []
    for i, (c, o) in enumerate(zip(card, orlm)):
        row = ours.get(mapping[i]["nl4opt_id"])
        our_obj = row["objective"] if row else None
        c_ans, o_ans = num(c["en_answer"]), num(o["en_answer"])
        rows.append({
            "cardinal_idx": i,
            "nl4opt_id": mapping[i]["nl4opt_id"],
            "question_edited": normalize(c["en_question"]) != normalize(o["en_question"]),
            "answer_changed": not same(c_ans, o_ans),
            "cardinal_answer": c["en_answer"],
            "orlm_answer": o["en_answer"],
            "ours": our_obj,
            "ours_status": row["status"] if row else "not converted",
            "in_our_mismatches": i in mismatch_idx,
            "ours_matches_cardinal": same(our_obj, c_ans) if c_ans is not None else None,
            "ours_matches_orlm": same(our_obj, o_ans) if o_ans is not None else (our_obj is None),
        })

    corrected = [r for r in rows if r["answer_changed"] or r["question_edited"]]
    summary = {
        "items": len(rows),
        "answer_changed": sum(r["answer_changed"] for r in rows),
        "question_edited": sum(r["question_edited"] for r in rows),
        "corrected_total": len(corrected),
        "our_mismatches": len(mismatch_idx),
        "mismatch_and_corrected": sorted(r["cardinal_idx"] for r in corrected if r["in_our_mismatches"]),
        "mismatch_not_corrected": sorted(i for i in mismatch_idx if not (rows[i]["answer_changed"] or rows[i]["question_edited"])),
        "corrected_not_mismatch": sorted(r["cardinal_idx"] for r in corrected if not r["in_our_mismatches"]),
        "ours_vs_orlm_scored": sum(1 for r in rows if num(r["orlm_answer"]) is not None),
        "ours_matches_orlm": sum(1 for r in rows if num(r["orlm_answer"]) is not None and r["ours_matches_orlm"]),
    }
    out = {"summary": summary, "corrected": corrected, "rows": rows}
    (PROC / "orlmbench_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
