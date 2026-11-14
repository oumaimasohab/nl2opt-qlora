"""Map each CardinalOperations/NL4OPT test item to its original NL4Opt id.

CardinalOperations items carry no id. Match on text: exact normalised match first,
then best 3-gram Jaccard over all NL4Opt splits. Writes data/processed/cardinal_test_ids.jsonl.

Run: uv run python -I scripts/02_map_test_ids.py
"""

import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from textsim import jaccard, load_cardinal, load_nl4opt, normalize, numbers, shingles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "cardinal_test_ids.jsonl"
FUZZY_MIN = 0.80  # below this a match is reported as unmapped

# Checked by hand: cardinal_idx -> NL4Opt test id. Item 28 is the test problem rewritten
# with its numbers moved into a "Concrete Data:" block (same numbers), so text match fails.
MANUAL = {28: "1117636837"}


def main():
    nl4opt = load_nl4opt(RAW / "nl4opt")
    cardinal = load_cardinal(RAW / "cardinal_nl4opt" / "NL4OPT_with_optimal_solution.json")
    by_norm = {}
    for r in nl4opt:
        by_norm.setdefault(normalize(r["document"]), []).append(r)
        r["sh"] = shingles(r["document"])

    out = []
    for i, c in enumerate(cardinal):
        q = c["en_question"]
        exact = by_norm.get(normalize(q), [])
        if i in MANUAL:
            best = next(r for r in nl4opt if r["id"] == MANUAL[i] and r["split"] == "test")
            score, method = jaccard(shingles(q), best["sh"]), "manual"
        elif exact:
            best, score, method = exact[0], 1.0, "exact"
            # Prefer a test-split hit if the text also appears elsewhere.
            best = next((r for r in exact if r["split"] == "test"), best)
        else:
            sh = shingles(q)
            best = max(nl4opt, key=lambda r: jaccard(sh, r["sh"]))
            score, method = jaccard(sh, best["sh"]), "fuzzy"
        mapped = method in ("exact", "manual") or score >= FUZZY_MIN
        out.append({
            "cardinal_idx": i,
            "nl4opt_id": best["id"] if mapped else None,
            "nl4opt_split": best["split"] if mapped else None,
            "method": method if mapped else "unmapped",
            "score": round(score, 4),
            "numbers_equal": numbers(q) == numbers(best["document"]),
            "also_in": sorted({r["split"] for r in exact} - {best["split"]}),
            "en_answer": c.get("en_answer"),
            # -99999 is CardinalOperations' placeholder, not an optimum; exclude these from scoring.
            "answer_sentinel": str(c.get("en_answer")).strip().startswith("-99999"),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    ids = [r["nl4opt_id"] for r in out if r["nl4opt_id"]]
    print(f"cardinal items: {len(cardinal)}")
    print("methods:", dict(Counter(r["method"] for r in out)))
    print("mapped split:", dict(Counter(r["nl4opt_split"] for r in out)))
    print("fuzzy with differing numbers:", sum(1 for r in out if r["method"] == "fuzzy" and not r["numbers_equal"]))
    print("duplicate ids:", len(ids) - len(set(ids)))
    print("sentinel answers (-99999):", [r["cardinal_idx"] for r in out if r["answer_sentinel"]])
    print("lowest scores:", sorted(r["score"] for r in out)[:8])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
