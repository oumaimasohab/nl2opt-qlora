"""Evaluate every checkpoint of a LoRA run on dev (zero-shot, no fixes) and log each to W&B.

Usage: uv run python scripts/08_eval_checkpoints.py adapters/v1 [--split dev] [--no-wandb]

Results go to results/finetune/<run>/<split>-ckpt<iteration>/; prints a table at the end.
Generations are reused if a checkpoint was already evaluated.
"""

import argparse
import json
from pathlib import Path

from nl2opt.adapters import checkpoints
from nl2opt.generate import DEFAULT_MODEL, MLXGenerator
from nl2opt.run_eval import EVAL_DIR, ROOT, evaluate, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", help="evaluate only these iterations")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter_dir)
    items = load_jsonl(EVAL_DIR / f"{args.split}.jsonl")
    rows = []
    for iteration, path in checkpoints(adapter_dir):
        if args.only and iteration not in args.only:
            continue
        out_dir = ROOT / "results" / "finetune" / adapter_dir.name / f"{args.split}-ckpt{iteration:04d}"
        print(f"== iteration {iteration}: {path}")
        generator = MLXGenerator(DEFAULT_MODEL, adapter_path=path)
        summary, _ = evaluate(
            items, generator, [], out_dir,
            config_extra={"model": DEFAULT_MODEL, "adapter": str(path), "split": args.split,
                          "eval_set": f"data/eval/{args.split}.jsonl", "iteration": iteration},
        )
        del generator
        if not args.no_wandb:
            from nl2opt.tracking import log_eval
            log_eval(path, iteration, args.split, 0, summary)
        rows.append((iteration, summary))

    print(f"\n{'iter':>5} {'accuracy':>9} {'runs':>6} {'optimal':>8}  counts")
    for iteration, s in rows:
        print(f"{iteration:>5} {s['accuracy']:>9.1%} {s['runs_rate']:>6.1%} {s['optimal_rate']:>8.1%}  "
              f"{ {k: v for k, v in s['counts'].items() if v} }")
    (ROOT / "results" / "finetune" / adapter_dir.name / f"{args.split}-checkpoints.json").write_text(
        json.dumps([{"iteration": i, **s} for i, s in rows], indent=2) + "\n")


if __name__ == "__main__":
    main()
