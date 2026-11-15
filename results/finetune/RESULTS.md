# Fine-tuning results: v1

QLoRA on `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`, config `configs/lora_v1.yaml`. All numbers below are greedy, zero-shot, using the same prompt and harness as `results/baselines/RESULTS.md`.

## Training

- **Data:** 690 NL4Opt train problems that the converter solves to Optimal (`data/sft/train.jsonl`). Validation uses 90 dev problems.
- **LoRA settings:** rank 16, scale 2.0, dropout 0.05, all 36 layers, all 7 projections.
- **Optimizer:** AdamW, lr 1e-4 with cosine decay and 20 warmup steps, batch 4.
- **Length:** 520 iterations, about 3 epochs. Prompt masked, gradient checkpointing on.
- **Cost:** about 70 minutes on an Apple M5 with 24 GB; peak MLX memory 6.21 GB. W&B run `v1`; the 50-iteration trial is `812vgqw8`.

| iter | val loss | dev acc (0-shot, no fixes) |
|---|---|---|
| 1 | 0.704 | — |
| 50 (trial run) | 0.049 | 46.7% (42/90) |
| 100 | 0.044 | 62.2% (56) |
| 200 | 0.038 | 75.6% (68) |
| 300 | **0.036** | 77.8% (70) |
| 400 | 0.038 | **80.0% (72)** |
| 500 | 0.039 | 80.0% (72) |
| 520 (final) | 0.039 | 80.0% (72) |

- **Loss and accuracy disagree after step 300.** Validation loss is lowest at 300, but dev accuracy keeps improving until 400. Loss averages over every token, including comments and wording. Accuracy only depends on getting the formulation right.
- **Checkpoints 400, 500 and 520 tie at 72 correct**, but they get different items right: 86 of 90 outcomes are the same.

**Selected checkpoint: 400**, the earliest of the tied best and the one with the lowest validation loss among them. Since 400 was chosen on dev, the dev figure is slightly optimistic. Test was run once, with that checkpoint only.

## Comparison

| setup | dev (90) | test (213) |
|---|---|---|
| base, 0-shot | 0.0% | 0.5% |
| base, 3-shot | 40.0% | 51.6% |
| base, 0-shot + imports/solver/operators fixes | 40.0% | 50.7% |
| base, 3-shot + imports/solver/operators fixes (previous best) | 53.3% | 55.9% |
| **v1 ckpt 400, 0-shot, no fixes** | **80.0%** | **84.5% (180)** |
| v1 ckpt 400, 0-shot + imports/solver/operators fixes | 82.2% | 84.5% |

Results are in `results/finetune/v1/{dev,test}-ckpt*/` and `results/finetune/v1-fix-*/`.

### What the fixes add after fine-tuning

- **Imports and solver fixes:** never triggered. The fine-tuned model always writes `import pulp` and calls the solver correctly, because every training target does.
- **Operators fix:** recovers 2 dev items that used `<` or `>` in a constraint, and none on test. Fine-tuning almost removed this error: test has 1 runtime error in total, a syntax error.

### Test failure breakdown (33 of 213)

| outcome | count |
|---|---|
| wrong_value | 27 |
| not_optimal | 5 |
| runtime_error (syntax) | 1 |

Before fine-tuning, most failures were code that didn't run. After fine-tuning, the code runs (99.5%) and solves to Optimal (97.2%). What's left are formulation errors: wrong coefficients, constraints or integrality.

## Caveats

1. **Train and test labels come from the same converter.** The model learns the converter's conventions:
   - integer vs continuous variables;
   - "exceeds" read as `>=`;
   - how percentage limits are written.

   The test references follow those same conventions. Test is restricted to the 213 items where the converter agrees with ORLMBenchmark, which guards against plain converter bugs. It doesn't remove the shared conventions, so part of the gain over the base model is learning those conventions, not only better modelling. The base model never saw them, except in the 3-shot exemplars.
2. **The 213-item test set is easier than the full test set.** It excludes the items the converter gets wrong, which are probably harder ones.
3. **One seed, one run.** With 213 items, one item is 0.47 points; the 95% interval on 84.5% is roughly ±5 points.

## Possible next steps

- **Error analysis on the 27 wrong-value test items.** Check how many are real model errors and how many are label ambiguity, such as strict inequalities or "must be" read as equality (see `converter.md`).
- **A shorter run.** Dev accuracy plateaus by step 400 (about 2.3 epochs), so about 400 iterations would do.
- **Few-shot plus adapter:** probably unnecessary, and it lengthens the prompt.
