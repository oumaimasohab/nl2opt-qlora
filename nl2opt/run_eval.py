"""Evaluate a model on an eval set: generate -> extract code -> run -> score.

Usage:
  uv run python -m nl2opt.run_eval --split dev --shots 0
  uv run python -m nl2opt.run_eval --split dev --shots 3 --limit 10
  uv run python -m nl2opt.run_eval --split dev --adapter adapters/v1/0000300_adapters.safetensors --wandb

Writes to results/<tag>/<split>-<shots>shot/:
  config.json        model, decoding, prompt settings, exemplar ids, versions
  generations.jsonl  raw model output per item (reused on re-run: generation resumes, never repeats)
  scores.jsonl       per item: extracted code, execution result, outcome
  summary.json       accuracy and outcome counts
"""

import argparse
import hashlib
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

from nl2opt.errors import categorize
from nl2opt.execute import DEFAULT_TIMEOUT, run_program
from nl2opt.extract import extract_code
from nl2opt.fixes import apply_fixes
from nl2opt.prompts import SYSTEM_PROMPT, build_messages
from nl2opt.score import ABS_TOL, REL_TOL, score, summarize

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "eval"


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rows):
    with Path(path).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_fingerprint(exemplars):
    """Changes whenever the system prompt or exemplars change; guards resumed runs."""
    blob = json.dumps({"system": SYSTEM_PROMPT, "exemplars": [[e["question"], e["program"]] for e in exemplars]})
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def versions():
    out = {"python": platform.python_version()}
    for pkg in ("mlx", "mlx-lm", "pulp", "highspy", "nl2opt"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def generate_all(items, generator, exemplars, gen_path, max_tokens, batch_size, log=print):
    """Generate for items without a saved generation. Appends to gen_path after every batch."""
    done = {g["id"] for g in load_jsonl(gen_path)}
    todo = [it for it in items if it["id"] not in done]
    if done:
        log(f"resuming: {len(done)} generations on disk, {len(todo)} to go")
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        t0 = time.time()
        outs = generator.generate([build_messages(it["question"], exemplars) for it in batch], max_tokens)
        seconds = time.time() - t0
        append_jsonl(gen_path, [{"id": it["id"], **out} for it, out in zip(batch, outs)])
        log(f"generated {min(start + batch_size, len(todo))}/{len(todo)} ({seconds:.1f}s for {len(batch)})")
    by_id = {g["id"]: g for g in load_jsonl(gen_path)}
    return [by_id[it["id"]] for it in items]


def score_all(items, generations, timeout, workers=8, fixes=()):
    """Extract, optionally fix, run and score every generation. ``fixes``: names from nl2opt.fixes."""

    def one(pair):
        item, gen = pair
        code, method = extract_code(gen["text"])
        if code is not None and fixes:
            code = apply_fixes(code, fixes)
        result = run_program(code, timeout=timeout) if code is not None else None
        outcome = score(code, result, item["reference"])
        return {
            "id": item["id"],
            "reference": item["reference"],
            "outcome": outcome,
            "extract_method": method,
            "status": result.status if result else None,
            "objective": result.objective if result else None,
            "objective_source": result.objective_source if result else None,
            "n_solves": result.n_solves if result else 0,
            "error": (result.error or "")[-1500:] if result else None,
            "error_category": categorize(result.error) if result else None,
            "truncated": gen.get("truncated"),
            "code": code,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, zip(items, generations)))


def summarize_scores(scores):
    """summarize() plus error categories and the stdout-fallback count."""
    summary = summarize([s["outcome"] for s in scores])
    categories = {}
    for s in scores:
        if s["outcome"] == "runtime_error":
            categories[s["error_category"]] = categories.get(s["error_category"], 0) + 1
    summary["runtime_error_categories"] = dict(sorted(categories.items(), key=lambda kv: -kv[1]))
    summary["objective_from_stdout"] = sum(1 for s in scores if s["objective_source"] == "stdout")
    return summary


def evaluate(items, generator, exemplars, out_dir, *, max_tokens=1024, batch_size=8,
             timeout=DEFAULT_TIMEOUT, config_extra=None, log=print):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "n_items": len(items),
        "shots": len(exemplars),
        "exemplar_ids": [e["id"] for e in exemplars],
        "prompt_fingerprint": prompt_fingerprint(exemplars),
        "max_tokens": max_tokens,
        "decoding": "greedy",
        "batch_size": batch_size,
        "exec_timeout_s": timeout,
        "tolerance": {"rel": REL_TOL, "abs": ABS_TOL},
        "versions": versions(),
        **(config_extra or {}),
    }
    config_path = out_dir / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("prompt_fingerprint", "max_tokens", "model", "adapter"):
            if old.get(key) != config.get(key):
                raise SystemExit(
                    f"{out_dir} holds a run with a different {key} ({old.get(key)!r} vs {config.get(key)!r}). "
                    "Use another --tag or delete the directory."
                )
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    t0 = time.time()
    generations = generate_all(items, generator, exemplars, out_dir / "generations.jsonl", max_tokens, batch_size, log)
    gen_seconds = time.time() - t0

    scores = score_all(items, generations, timeout)
    (out_dir / "scores.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in scores))

    summary = summarize_scores(scores)
    summary.update({
        "truncated": sum(1 for g in generations if g.get("truncated")),
        "mean_generation_tokens": round(sum(g["generation_tokens"] for g in generations) / max(len(generations), 1), 1),
        "generation_seconds_this_run": round(gen_seconds, 1),
    })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--shots", type=int, choices=(0, 1, 2, 3), default=0)
    ap.add_argument("--limit", type=int, help="evaluate only the first N items (smoke runs)")
    ap.add_argument("--model", default=None, help="mlx model id (default: Qwen2.5-Coder-3B-Instruct-4bit)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter directory or checkpoint file")
    ap.add_argument("--wandb", action="store_true", help="log the summary to the adapter's W&B training run")
    ap.add_argument("--name", default=None, help="run directory name (default: <split>-<shots>shot)")
    ap.add_argument("--tag", default="baselines", help="results/<tag>/...")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    from nl2opt.generate import DEFAULT_MODEL, MLXGenerator

    items = load_jsonl(EVAL_DIR / f"{args.split}.jsonl")
    if not items:
        raise SystemExit(f"no eval set at {EVAL_DIR / (args.split + '.jsonl')}; run scripts/06_make_eval_sets.py")
    if args.limit:
        items = items[: args.limit]
    exemplars = load_jsonl(EVAL_DIR / "fewshot.jsonl")[: args.shots]
    model_id = args.model or DEFAULT_MODEL

    suffix = f"-limit{args.limit}" if args.limit else ""
    out_dir = ROOT / "results" / args.tag / (args.name or f"{args.split}-{args.shots}shot{suffix}")
    generator = MLXGenerator(model_id, adapter_path=args.adapter)
    summary, _ = evaluate(
        items, generator, exemplars, out_dir,
        max_tokens=args.max_tokens, batch_size=args.batch_size, timeout=args.timeout,
        config_extra={"model": model_id, "adapter": args.adapter, "split": args.split,
                      "eval_set": f"data/eval/{args.split}.jsonl", "limit": args.limit},
    )
    print(json.dumps(summary, indent=2))
    print(f"results in {out_dir}")
    if args.wandb:
        if not args.adapter:
            raise SystemExit("--wandb needs --adapter (results are logged to that adapter's training run)")
        from nl2opt.adapters import iteration_of
        from nl2opt.tracking import log_eval

        run_id = log_eval(args.adapter, iteration_of(args.adapter), args.split, args.shots, summary)
        print(f"logged to W&B run {run_id}")


if __name__ == "__main__":
    main()
