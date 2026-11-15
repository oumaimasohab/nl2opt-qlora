"""Log evaluation results into the W&B run that mlx_lm.lora created for an adapter.

mlx_lm.lora logs training metrics with step = training iteration. Evaluations happen afterwards and
out of order, so they use their own x-axis (eval/iteration) instead of the W&B step.
"""

import json
from pathlib import Path

from nl2opt.adapters import adapter_root


def find_run_id(adapter_dir):
    """W&B run id from <adapter_dir>/wandb/run-<date>-<id> (newest), or None."""
    runs = sorted((Path(adapter_dir) / "wandb").glob("run-*-*"))
    return runs[-1].name.rsplit("-", 1)[-1] if runs else None


def log_eval(adapter_path, iteration, split, shots, summary, fixes=()):
    import wandb

    root = adapter_root(adapter_path)
    config = json.loads((root / "adapter_config.json").read_text())
    run_id = find_run_id(root)
    if run_id is None:
        raise RuntimeError(f"no wandb run under {root}/wandb; was training run with report_to: wandb?")
    prefix = f"eval/{split}-{shots}shot" + (f"+{'+'.join(fixes)}" if fixes else "")
    run = wandb.init(project=config.get("project_name"), id=run_id, resume="must", dir=str(root))
    try:
        run.define_metric("eval/iteration")
        run.define_metric("eval/*", step_metric="eval/iteration")
        run.log({
            "eval/iteration": iteration,
            f"{prefix}/accuracy": summary["accuracy"],
            f"{prefix}/runs_rate": summary["runs_rate"],
            f"{prefix}/optimal_rate": summary["optimal_rate"],
            **{f"{prefix}/count/{k}": v for k, v in summary["counts"].items()},
        })
    finally:
        run.finish()
    return run_id
