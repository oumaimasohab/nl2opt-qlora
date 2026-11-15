"""Build the evaluation sets and the few-shot exemplars.

  data/eval/dev.jsonl       NL4Opt dev problems the converter solves to Optimal (reference = converter optimum)
  data/eval/test.jsonl      ORLMBenchmark items whose corrected answer equals the converter's optimum
                            (two independent sources agree), with ORLMBenchmark's question text
  data/eval/fewshot.jsonl   3 train exemplars (problem + converted program), see nl2opt.prompts

Run: uv run python -I scripts/06_make_eval_sets.py
"""

import json
from pathlib import Path

from nl2opt.convert import parse
from nl2opt.prompts import select_exemplars
from nl2opt.score import parse_reference

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "eval"


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dev_set():
    rows = load_jsonl(PROC / "nl4opt_dev.pulp.jsonl")
    return [
        {"id": f"dev:{r['id']}", "nl4opt_id": r["id"], "question": r["document"],
         "reference": r["objective"], "reference_source": "converter"}
        for r in rows if r["status"] == "Optimal"
    ]


def test_set():
    orlm = load_jsonl(RAW / "orlmbench" / "NL4OPT.jsonl")
    comparison = json.loads((PROC / "orlmbench_comparison.json").read_text())["rows"]
    items = []
    for row in comparison:
        ref = parse_reference(row["orlm_answer"])
        if ref is None or not row["ours_matches_orlm"]:
            continue
        i = row["cardinal_idx"]
        items.append({
            "id": f"test:{i}", "cardinal_idx": i, "nl4opt_id": row["nl4opt_id"],
            "question": orlm[i]["en_question"], "reference": ref,
            "reference_source": "orlmbench (agrees with converter)",
            "question_edited_by_orlm": row["question_edited"],
        })
    return items


def fewshot():
    raw = {}
    for line in (PROC / "nl4opt_train.dedup.jsonl").read_text(encoding="utf-8").splitlines():
        (key, rec), = json.loads(line).items()
        raw[str(key)] = rec
    candidates = []
    for r in load_jsonl(PROC / "nl4opt_train.pulp.jsonl"):
        model = parse(raw[r["id"]])
        candidates.append({
            **r,
            "kinds": sorted({c["kind"] for c in model["constraints"]}),
            "all_integer": len(r["integer_vars"]) == len(r["vars"]),
            "all_continuous": not r["integer_vars"],
        })
    chosen = select_exemplars(candidates, seed=0)
    return [
        {"id": c["id"], "slot": c["slot"], "question": c["document"], "program": c["program"],
         "objective": c["objective"], "kinds": c["kinds"]}
        for c in chosen
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("dev", dev_set()), ("test", test_set()), ("fewshot", fewshot())):
        write_jsonl(OUT / f"{name}.jsonl", rows)
        print(f"wrote {OUT / (name + '.jsonl')} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
