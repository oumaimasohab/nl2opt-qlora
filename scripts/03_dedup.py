"""Deduplicate NL4Opt train/dev against the test set and within themselves.

Drops a train/dev problem if it
  1. has an NL4Opt id that is in the test split or the CardinalOperations test mapping,
  2. has the same normalised text as a test problem, or
  3. has word-3-gram Jaccard >= --near with any test problem (near-duplicate),
and drops later copies of exact / near duplicates inside train+dev (train is kept first).

Same-template problems with different entities and numbers score about 0.3-0.5 and are kept;
they are listed under "template_overlap" in the report. Use --near 0.3 to drop them as well.

Writes data/processed/nl4opt_{train,dev}.dedup.jsonl (original record format) and dedup_report.json.
Run: uv run python -I scripts/03_dedup.py [--near 0.8]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsim import jaccard, load_nl4opt, normalize, shingles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "nl4opt"
PROC = ROOT / "data" / "processed"
TEMPLATE_INFO = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--near", type=float, default=0.8, help="3-gram Jaccard threshold for near-duplicates")
    args = ap.parse_args()

    rows = load_nl4opt(RAW)
    for r in rows:
        r["norm"] = normalize(r["document"])
        r["sh"] = shingles(r["document"])
    test = [r for r in rows if r["split"] == "test"]
    cand = [r for r in rows if r["split"] in ("train", "dev")]  # train first, then dev

    mapping = [json.loads(l) for l in (PROC / "cardinal_test_ids.jsonl").read_text().splitlines()]
    test_ids = {r["id"] for r in test} | {m["nl4opt_id"] for m in mapping if m["nl4opt_id"]}
    test_norm = {r["norm"] for r in test}

    dropped, kept, template_overlap = [], [], []
    for r in cand:
        best_s, best_t = max(((jaccard(r["sh"], t["sh"]), t) for t in test), key=lambda x: x[0])
        reason = None
        if r["id"] in test_ids:
            reason = "id_in_test"
        elif r["norm"] in test_norm:
            reason = "exact_text_in_test"
        elif best_s >= args.near:
            reason = "near_dup_of_test"
        else:
            for k in kept:
                if r["norm"] == k["norm"] or jaccard(r["sh"], k["sh"]) >= args.near:
                    reason, best_t, best_s = "dup_within_train_dev", k, jaccard(r["sh"], k["sh"])
                    break
        if reason:
            dropped.append({"id": r["id"], "split": r["split"], "reason": reason,
                            "match_id": best_t["id"], "match_split": best_t["split"], "score": round(best_s, 4)})
            continue
        kept.append(r)
        if best_s >= TEMPLATE_INFO:
            template_overlap.append({"id": r["id"], "split": r["split"], "test_id": best_t["id"], "score": round(best_s, 4)})

    PROC.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "dev"):
        out = [r for r in kept if r["split"] == split]
        with (PROC / f"nl4opt_{split}.dedup.jsonl").open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps({r["id"]: r["record"]}, ensure_ascii=False) + "\n")
        counts[split] = {"before": sum(1 for r in cand if r["split"] == split), "after": len(out)}

    report = {
        "near_threshold": args.near,
        "test_size_nl4opt": len(test),
        "test_size_cardinal": len(mapping),
        "counts": counts,
        "dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
        "dropped": dropped,
        "template_overlap_threshold": TEMPLATE_INFO,
        "template_overlap": sorted(template_overlap, key=lambda x: -x["score"]),
    }
    (PROC / "dedup_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("near_threshold", "counts", "dropped_by_reason")}, indent=2))
    print(f"template overlap (kept, 3g >= {TEMPLATE_INFO}): {len(template_overlap)}")


if __name__ == "__main__":
    main()
