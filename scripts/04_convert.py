"""Convert every NL4Opt formulation to a PuLP program, run each program, and validate.

For each record in train/dev/test:
  - parse the formulation and decide integer variables (nl2opt/convert.py),
  - render a PuLP + HiGHS program, twice: share-of-total constraints simplified
    (0.7*x - 0.3*y <= 0, field "program") and as stated (x <= 0.3*(x + y), "program_literal"),
  - run the program in a fresh interpreter and check its printed objective equals an
    in-process solve of the same model (catches rendering bugs).
Then compare test objectives with the CardinalOperations/NL4OPT reference optima.

Writes:
  data/processed/nl4opt_{train,dev,test}.pulp.jsonl   one row per problem (train/dev are the deduped sets)
  data/processed/converter_report.json                validation summary

Run: uv run python -I scripts/04_convert.py
"""

import json
import math
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nl2opt.convert import IncompleteFormulation, parse, solve, to_program

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nl4opt"
PROC = ROOT / "data" / "processed"
SOURCES = {
    "train": PROC / "nl4opt_train.dedup.jsonl",
    "dev": PROC / "nl4opt_dev.dedup.jsonl",
    "test": RAW / "test.jsonl",
}
TOL = dict(rel_tol=1e-4, abs_tol=1e-4)


def run_program(code):
    """Execute a generated program; return (returncode, status, objective, stderr tail)."""
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    status = re.search(r"^Status: (.+)$", r.stdout, re.M)
    obj = re.search(r"^Objective: (.+)$", r.stdout, re.M)
    try:
        value = float(obj.group(1)) if obj else None
    except ValueError:
        value = None
    return r.returncode, status.group(1) if status else None, value, r.stderr.strip()[-300:]


def convert_split(split):
    rows, skipped = [], []
    with SOURCES[split].open(encoding="utf-8") as f:
        for line in f:
            (key, rec), = json.loads(line).items()
            try:
                model = parse(rec)
            except IncompleteFormulation as e:
                skipped.append({"id": str(key), "reason": str(e)})
                continue
            status, objective = solve(model)
            rows.append({
                "id": str(key),
                "split": split,
                "document": rec["document"],
                "vars": model["vars"],
                "integer_vars": model["integer"],
                "program": to_program(model),
                # Same model with share-of-total constraints written as stated: x <= 0.3 * (x + y).
                "program_literal": to_program(model, ratio_style="literal"),
                "status": status,
                "objective": objective,
            })
    with ThreadPoolExecutor(max_workers=8) as ex:
        runs = list(ex.map(lambda r: run_program(r["program"]), rows))
        runs_literal = list(ex.map(lambda r: run_program(r["program_literal"]), rows))
    problems = []
    checks = [("program", row, run) for row, run in zip(rows, runs)]
    checks += [("program_literal", row, run) for row, run in zip(rows, runs_literal)]
    for field, row, (rc, st, val, err) in checks:
        # Objectives are only meaningful for Optimal solves; otherwise compare status alone.
        same = st == row["status"] and (
            st != "Optimal" or (val is not None and math.isclose(val, row["objective"], **TOL))
        )
        if rc != 0 or not same:
            problems.append({"id": row["id"], "field": field, "returncode": rc, "program_status": st,
                             "program_objective": val, "model_objective": row["objective"], "stderr": err})
    with (PROC / f"nl4opt_{split}.pulp.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows, problems, skipped


def cardinal_agreement(test_rows):
    by_id = {r["id"]: r for r in test_rows}
    mapping = [json.loads(l) for l in (PROC / "cardinal_test_ids.jsonl").read_text().splitlines()]
    scored = [m for m in mapping if not m["answer_sentinel"]]
    out = Counter()
    mismatches = []
    for m in scored:
        row = by_id.get(m["nl4opt_id"])
        ref = float(m["en_answer"])
        if row is None:
            out["skipped"] += 1
            mismatches.append({"cardinal_idx": m["cardinal_idx"], "nl4opt_id": m["nl4opt_id"],
                               "reference": ref, "ours": None, "note": "annotation incomplete, not converted"})
            continue
        ok = row["objective"] is not None and math.isclose(row["objective"], ref, **TOL)
        out["match"] += ok
        if not ok:
            mismatches.append({"cardinal_idx": m["cardinal_idx"], "nl4opt_id": m["nl4opt_id"],
                               "reference": ref, "ours": row["objective"], "integer_vars": row["integer_vars"]})
    return {"scored": len(scored), "excluded_sentinel": len(mapping) - len(scored),
            "match": out["match"], "match_rate": round(out["match"] / len(scored), 4), "mismatches": mismatches}


def main():
    report = {"splits": {}}
    all_rows = {}
    for split in SOURCES:
        rows, problems, skipped = convert_split(split)
        all_rows[split] = rows
        report["splits"][split] = {
            "problems": len(rows),
            "skipped_incomplete": skipped,
            "status": dict(Counter(r["status"] for r in rows)),
            "with_integer_vars": sum(1 for r in rows if r["integer_vars"]),
            "all_vars_integer": sum(1 for r in rows if len(r["integer_vars"]) == len(r["vars"])),
            "all_vars_continuous": sum(1 for r in rows if not r["integer_vars"]),
            "program_run_failures": problems,
        }
        s = report["splits"][split]
        print(f"{split}: {s['problems']} programs ({len(skipped)} skipped: incomplete annotation), status {s['status']}, "
              f"all-int {s['all_vars_integer']}, all-cont {s['all_vars_continuous']}, "
              f"run failures/mismatches {len(problems)}")
    report["cardinal_test"] = cardinal_agreement(all_rows["test"])
    c = report["cardinal_test"]
    print(f"CardinalOperations agreement: {c['match']}/{c['scored']} ({c['match_rate']:.1%}), "
          f"{c['excluded_sentinel']} placeholder answers excluded")
    (PROC / "converter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {PROC / 'converter_report.json'}")


if __name__ == "__main__":
    main()
