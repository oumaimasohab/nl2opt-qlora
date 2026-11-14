# Smoke test results

Run on 2026-11-14 against commit `1338de8` plus uncommitted `smoke/`, `data/`, `adapters/`.

## Environment and versions

| Item | Value |
|---|---|
| Machine | Apple M5, 24 GB unified memory (MLX recommended working set 19.07 GB) |
| macOS | 27.2.1 |
| Python | 3.12.12 (uv venv) |
| mlx | 0.32.4 |
| mlx-lm | 0.32.2 |
| transformers | 5.23.0 |
| PuLP | 3.3.2 |
| highspy / HiGHS | 1.16.0 |
| CBC | bundled with PuLP (`PULP_CBC_CMD`) |
| Model | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` |

## 1. Solver check (`smoke/01_solver.py`)

Toy LP: max 3x + 5y with x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18. The expected optimum is 36 at (2, 6).

Available solvers: `['PULP_CBC_CMD', 'HiGHS']`

| Solver | Status | Objective | x | y | Result |
|---|---|---|---|---|---|
| PULP_CBC_CMD | Optimal | 36.0 | 2.0 | 6.0 | PASS |
| HiGHS | Optimal | 36.0 | 2.0 | 6.0 | PASS |

The program the base model generated for the bakery problem (`smoke/test.py`) also runs:
**Status: Optimal**, profit 1350.0 (90 chocolate cakes, 30 vanilla cakes). It uses PuLP's default solver (CBC), not HiGHS as the prompt asked.

## 2. Inference speed and memory (`smoke/02_generate.py`)

This uses the base model with no adapter. The prompt is 258 tokens, `max_tokens=1024`, and the output is 365 tokens. Run 1 is a warm-up and is discarded.

| Run | Prompt tok/s | Generation tok/s | Peak memory |
|---|---|---|---|
| verbose `generate` | 809.7 | 60.8 | 2.14 GB |
| 1 (warm-up) | 1276.6 | 61.2 | 2.14 GB |
| 2 | 1238.3 | 60.9 | 2.14 GB |
| 3 | 1266.0 | 61.0 | 2.14 GB |
| **Recorded (mean of 2–3)** | **1252.2** | **60.9** | **2.14 GB** |

That works out to about **6.0 s per sample** at roughly 365 output tokens.

## 3. LoRA training (`mlx_lm.lora`, config in `adapters/smoke/adapter_config.json`)

`
uv run mlx_lm.lora \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --train \
  --data data/smoke \
  --iters 10 \
  --batch-size 2 \
  --mask-prompt \
  --adapter-path adapters/smoke \
  --steps-per-eval 5 \
  --val-batches 1 \
  --steps-per-report 1
`

Settings: LoRA rank 8, scale 20, dropout 0, 16 layers, Adam with lr 1e-5, batch size 2, 10 iterations, `mask_prompt`, max sequence length 2048, seed 0.
Data: `data/smoke`, with 8 train and 2 valid examples. Trainable parameters: 6.652M of 3085.939M (0.216%).

| Metric | Value |
|---|---|
| Training throughput | ~337–391 tok/s (mean ≈ 364) |
| Tokens trained | 4.2k |
| Peak memory (MLX, training) | **5.66 GB** |
| Wall time (including model load) | ~15 s |

### Loss curve

These are the losses from training the current `adapters/smoke` on the fixed data.

| Iter | Train loss | Val loss | tok/s |
|---|---|---|---|
| 1 | 0.763 | 0.786 | 337 |
| 2 | 0.636 | | 357 |
| 3 | 0.486 | | 362 |
| 4 | 0.339 | | 379 |
| 5 | 0.222 | 0.228 | 379 |
| 6 | 0.132 | | 356 |
| 7 | 0.090 | | 359 |
| 8 | 0.047 | | 358 |
| 9 | 0.034 | | 359 |
| 10 | 0.023 | 0.026 | 391 |

```mermaid
xychart-beta
    title "Smoke LoRA loss"
    x-axis "iteration" [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "loss" 0 --> 0.9
    line [0.763, 0.636, 0.486, 0.339, 0.222, 0.132, 0.090, 0.047, 0.034, 0.023]
```

Train loss falls from 0.763 to 0.023 and validation loss from 0.786 to 0.026, so the training loop works. With 8 near-identical templated examples, this is memorisation, not generalisation.

For comparison, the earlier adapter trained on the broken `add_variable` data started higher, at a train loss of 1.002 (val 1.026). It finished at 0.037 (val 0.040). Valid PuLP starts closer to what the base model already writes, so the starting loss is lower.

### Adapter check (`smoke/05_check_adapter.py`)

| Check | Emits `# SMOKE-OK` | Want |
|---|---|---|
| Base model, train prompt | No | No ✅ |
| Adapter, train prompt | Yes | Yes ✅ |
| Adapter, valid prompt (unseen context) | Yes | info only |

**PASS.** The adapter changes the model's behaviour.
