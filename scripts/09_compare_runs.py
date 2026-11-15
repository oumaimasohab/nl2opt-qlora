"""Compare eval runs on the subset of problems with a share-of-total constraint vs the rest.

A problem is in the "share" subset when its converter reference contains a share-of-total
constraint ("x at most 30% of all vehicles") or a weighted-average constraint.

Usage:
  uv run python -I scripts/09_compare_runs.py --split test \
      results/finetune/v1/test-ckpt0400 results/finetune/v2/test-ckpt0500

Prints accuracy per subset for each run, and for the first two runs the items that flipped.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def share_ids(split):
    items = load(ROOT / "data" / "eval" / f"{split}.jsonl")
    labels = {r["id"]: r for r in load(ROOT / "data" / "processed" / f"nl4opt_{split}.pulp.jsonl")}
    return {it["id"] for it in items
            if "of total (" in labels[it["nl4opt_id"]]["program"]
            or "weighted average" in labels[it["nl4opt_id"]]["program"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("runs", nargs="+", help="result directories holding scores.jsonl")
    args = ap.parse_args()

    share = share_ids(args.split)
    correct = {}
    print(f"{'run':55s} {'share':>12s} {'other':>12s} {'all':>12s}")
    for run in args.runs:
        scores = {s["id"]: s["outcome"] == "correct" for s in load(Path(run) / "scores.jsonl")}
        correct[run] = scores
        cells = []
        for subset in (share, set(scores) - share, set(scores)):
            ok = sum(scores[i] for i in subset)
            cells.append(f"{ok}/{len(subset)} {ok / len(subset):.0%}")
        print(f"{run:55s} {cells[0]:>12s} {cells[1]:>12s} {cells[2]:>12s}")

    if len(args.runs) >= 2:
        a, b = (correct[r] for r in args.runs[:2])
        for subset_name, subset in (("share", share), ("other", set(a) - share)):
            gained = sorted(i for i in subset if b[i] and not a[i])
            lost = sorted(i for i in subset if a[i] and not b[i])
            print(f"{subset_name}: second run fixes {len(gained)} {gained}")
            print(f"{subset_name}: second run breaks {len(lost)} {lost}")


if __name__ == "__main__":
    main()
