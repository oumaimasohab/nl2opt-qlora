# nl2opt-qlora

This repo fine-tunes a small code model, Qwen2.5-Coder-3B-Instruct (4-bit), with QLoRA on a laptop using mlx-lm. The model learns to turn linear-programming word problems from NL4Opt into runnable PuLP programs. Scoring works by **running** each generated program and checking that it reaches the reference optimal objective.

```
"A boat carries 10 ducks per trip, a canoe 8 ... at least 60% of trips must be by canoe ...
 minimize total time."
        │  model
        ▼
prob = pulp.LpProblem(...);  boat = pulp.LpVariable(...);  prob += canoe >= 0.6 * (boat + canoe) ...
        │  run (subprocess, 30 s timeout)
        ▼
Optimal, objective 1160  ──  matches reference?  ✔ / ✘
```

## Results

Accuracy is the share of problems whose program runs, solves to Optimal, and matches the reference objective within 1e-4. All runs use greedy decoding and a single generation per problem.

| Setup | Dev (90) | Test (213) |
|---|---|---|
| Qwen2.5-Coder-3B, 0-shot | 0.0% | 0.5% |
| Qwen2.5-Coder-3B, 3-shot | 40.0% | 51.6% |
| Qwen2.5-Coder-3B, 3-shot + code fixes (imports, solver, `<`→`<=`) | 53.3% | 55.9% |
| **+ QLoRA v1**, 0-shot | 80.0% | 84.5% |
| **+ QLoRA v2**, 0-shot | **84.4%** | **92.5%** |

- **v1:** about 3 epochs over 690 NL4Opt train problems. Rank 16, all layers; roughly 70 min on an Apple M5 with 24 GB, peak memory 6.2 GB.
- **v2:** the same run, with one change to how 72 of the training targets write percentage constraints. "At most 30% of vehicles are bikes" becomes `bikes <= 0.3 * (bikes + scooters)` instead of the pre-simplified `0.7 * bikes - 0.3 * scooters <= 0`. Error analysis of v1 found this was half of its test failures. On test problems with such a constraint, accuracy rose from 62% to 92%; the others stayed at 92%. McNemar p = 0.002.

Details: [`results/baselines/RESULTS.md`](results/baselines/RESULTS.md) · [`results/finetune/RESULTS.md`](results/finetune/RESULTS.md) · [`results/finetune/ERRORS.md`](results/finetune/ERRORS.md)

**Caveats.**
- **Shared conventions.** Training targets and references both come from this repo's converter (below). The model learns the converter's conventions, such as which variables are integers.
- **Restricted test set.** Test is limited to the 213 problems where the converter agrees with ORLMBenchmark's corrected answers, so it likely leaves out harder problems.
- **One seed per run.** The 95% interval on 92.5% is about ±4 points.
- **v2 test isn't fully held out.** v2's change was designed after reading v1's test failures. The dev results, which played no part in that design, show the same effect.

## How it works

1. **Data** (`scripts/01_download.sh`). NL4Opt generation data (713 train / 99 dev / 289 test) carries structured formulations, not code. CardinalOperations/NL4OPT and ORLMBenchmark provide reference optima for the test problems. Every download is pinned to a commit and checked by sha256.
2. **Converter** (`nl2opt/convert.py`, notes in [`converter.md`](converter.md)). It turns each NL4Opt formulation into a PuLP program and decides integer vs continuous variables from the wording. Every program is run and checked against an in-process solve. On test it agrees with CardinalOperations on 200/223 answers.
3. **Eval sets** (`scripts/02`–`06`). `05` audits the converter against ORLMBenchmark.
   - dev = 90 dev problems the converter solves to Optimal;
   - test = 213 ORLMBenchmark items whose answer equals the converter's;
   - train/dev were checked against test for near-duplicates. None scored above 0.8 similarity; 24 template-level overlaps are listed in `data/processed/dedup_report.json`.
4. **Harness** (`nl2opt/run_eval.py`). It runs generate → extract the code block → execute in a subprocess → record the `LpProblem.solve` result → score. Generation resumes where it stopped, and every run saves its config, raw generations, per-item scores and a summary under `results/`.

   The harness doesn't sandbox anything: generated code runs with your user's permissions.
5. **Fine-tuning** (`scripts/07_make_sft_data.py`, `configs/lora_v*.yaml`). mlx-lm LoRA on the 4-bit model, using the same prompt as evaluation and loss on the answer only. Checkpoints are scored on dev (`scripts/08_eval_checkpoints.py`). The earliest best one is then run once on test.

## Reproduce

Needs Apple silicon (MLX) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                         # Python 3.12, mlx, mlx-lm, PuLP, HiGHS (pinned)
bash scripts/01_download.sh                     # -> data/raw/ (not in git)
for s in 02_map_test_ids 03_dedup 04_convert 05_compare_orlmbench 06_make_eval_sets; do uv run python -I scripts/$s.py; done
uv run python -I scripts/07_make_sft_data.py --ratio-style literal --out data/sft-v2

uv run python -m nl2opt.run_eval --split dev --shots 3             # base-model baseline
uv run python scripts/train_with_peak.py --config configs/lora_v2.yaml   # ~70 min; logs to W&B if logged in
uv run python scripts/08_eval_checkpoints.py adapters/v2           # dev accuracy per checkpoint
uv run python -m nl2opt.run_eval --split test --adapter adapters/v2/0000500_adapters.safetensors \
    --tag finetune/v2 --name test-ckpt0500

uv run pytest                                   # 143 tests, about 10 s
```

- **Optional scripts.** `scripts/09_compare_runs.py` breaks accuracy down by constraint type. `uv run python -m nl2opt.rescore <run> --fix imports` re-scores saved generations with a code fix applied.
- **W&B.** W&B logging is set by `report_to: wandb` in the config; remove that line to train without it.
- **Data and weights aren't in git.** Data comes from the download and the scripts. The adapters, about 120 MB per checkpoint, come from training.

## Layout

```
nl2opt/        converter, prompts, generation, execution, scoring, fixes, W&B logging
scripts/       numbered pipeline steps (01–09)
configs/       LoRA training configs
results/       every eval run: config, generations, scores, summary + RESULTS.md write-ups
tests/         pytest suite
smoke/         first end-to-end smoke test (solver, inference, a tiny LoRA run)
```

## Data licenses

- NL4Opt: see [nl4opt-competition](https://github.com/nl4opt/nl4opt-competition).
- CardinalOperations/NL4OPT: CC-BY-NC-4.0, used only for test reference answers, not for training.
- ORLMBenchmark: Apache-2.0.

No dataset files are committed. The code is Apache-2.0 (`LICENSE`).
