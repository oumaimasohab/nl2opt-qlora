"""Re-run and re-score a finished run's saved generations, optionally with code fixes applied.

The model is not loaded and nothing is regenerated, so any difference from the source run is caused
only by the fixes.

Usage:
  uv run python -m nl2opt.rescore results/baselines/test-0shot --fix imports
  uv run python -m nl2opt.rescore results/baselines/dev-3shot --fix imports --tag fix-imports

Writes results/<tag>/<source run name>/ with config.json, scores.jsonl, summary.json and
changes.jsonl (items whose outcome differs from the source run).
"""

import argparse
import json
from pathlib import Path

from nl2opt.errors import categorize
from nl2opt.execute import DEFAULT_TIMEOUT
from nl2opt.fixes import FIXES
from nl2opt.run_eval import ROOT, load_jsonl, score_all, summarize_scores


def rescore(source_dir, out_dir, fixes, timeout=DEFAULT_TIMEOUT, eval_path=None):
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    config = json.loads((source_dir / "config.json").read_text())
    eval_path = Path(eval_path or ROOT / config["eval_set"])
    by_id = {it["id"]: it for it in load_jsonl(eval_path)}
    generations = load_jsonl(source_dir / "generations.jsonl")
    items = [by_id[g["id"]] for g in generations]

    scores = score_all(items, generations, timeout, fixes=tuple(fixes))
    summary = summarize_scores(scores)
    summary["mean_generation_tokens"] = round(sum(g["generation_tokens"] for g in generations) / max(len(generations), 1), 1)

    source_scores = {s["id"]: s for s in load_jsonl(source_dir / "scores.jsonl")}
    changes = [
        {"id": s["id"], "before": source_scores[s["id"]]["outcome"], "after": s["outcome"],
         "error_before": categorize(source_scores[s["id"]].get("error")),
         "error_after": s["error_category"]}
        for s in scores if source_scores.get(s["id"], {}).get("outcome") != s["outcome"]
    ]
    source_summary = json.loads((source_dir / "summary.json").read_text())
    summary["source_accuracy"] = source_summary["accuracy"]
    summary["accuracy_delta"] = round(summary["accuracy"] - source_summary["accuracy"], 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {**config, "rescored_from": str(source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir),
         "fixes": list(fixes), "exec_timeout_s": timeout}, indent=2) + "\n")
    (out_dir / "scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))
    (out_dir / "changes.jsonl").write_text("".join(json.dumps(c) + "\n" for c in changes))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, changes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="finished run directory, e.g. results/baselines/test-0shot")
    ap.add_argument("--fix", action="append", default=[], choices=sorted(FIXES), help="repeatable")
    ap.add_argument("--tag", default=None, help="results/<tag>/... (default: fix-<fixes>)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    source = Path(args.source).resolve()
    tag = args.tag or ("fix-" + "+".join(args.fix) if args.fix else "rescore")
    out_dir = ROOT / "results" / tag / source.name
    summary, changes = rescore(source, out_dir, args.fix, timeout=args.timeout)
    print(json.dumps(summary, indent=2))
    print(f"{len(changes)} items changed outcome; results in {out_dir}")


if __name__ == "__main__":
    main()
