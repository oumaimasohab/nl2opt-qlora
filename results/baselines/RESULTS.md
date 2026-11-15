# Baselines: base model, zero-shot and 3-shot

Model: `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` with no adapter. Run on 2026-11-15.

## Setup

| Item | Value |
|---|---|
| Harness | `uv run python -m nl2opt.run_eval --split {dev,test} --shots {0,3}` |
| System prompt | Same as `smoke/02_generate.py` (`nl2opt.prompts.SYSTEM_PROMPT`) |
| Decoding | Greedy, max 1024 new tokens, `batch_generate` with batch size 8. Batched output is identical to unbatched (checked on 4 items). |
| 3-shot examples | `data/eval/fewshot.jsonl`: 3 fixed train problems with their converted programs. Train ids `1496462806` (ratio constraint, integer), `1434993476` (xby constraint, integer), `-1176378527` (continuous acres). |
| Execution | Subprocess with a 30 s timeout, in a fresh temp directory. Status and objective come from the recorded `LpProblem.solve` call, not from printed output. |
| Scoring | Strict: the program must run without error and reach `Optimal`, and the objective must match within rel/abs 1e-4. |
| Versions | mlx 0.32.4, mlx-lm 0.32.2, PuLP 3.3.2, HiGHS 1.16.0, Python 3.12.12 |

**Eval sets** (`scripts/06_make_eval_sets.py`):
- **dev:** 90 NL4Opt dev problems the converter solves to Optimal. The reference is the converter's optimum, so dev measures agreement with *our* labels. Use it to compare setups, not as a headline number.
- **test:** 213 ORLMBenchmark items whose corrected answer equals the converter's optimum. It uses ORLMBenchmark's question text.

## Dev results (n = 90)

| Setup | Accuracy | Code found | Runs | Optimal | Wrong value | Runtime error | Not optimal | Tokens/answer | Generation time |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | **0.0%** (0) | 100% | 1.1% | 1.1% | 1 | 89 | 0 | 272 | 148 s |
| 3-shot | **40.0%** (36) | 100% | 83.3% | 82.2% | 38 | 15 | 1 | 241 | 290 s |

No answer was truncated or timed out, and none needed the printed-output fallback.

### What goes wrong

**Zero-shot: almost every program crashes.** The model writes `from pulp import LpProblem, ...` and then uses names it never imported.

| Errors | Cause |
|---|---|
| 58 | `NameError`: `LpStatus`, `value` or `LpMinimize` used without importing them |
| 17 | Solver passed as a string, e.g. `prob.solve("HiGHS")` |
| 10 | Strict `<` or `>` between variables, which PuLP doesn't support |

55 of the 89 crashing programs had already solved the model before crashing, usually while printing the result. 27 of those had the correct optimum. A lenient scorer that took the recorded solve despite the crash would give zero-shot **27/90 (30%)**. We report strict scoring, because the task is a runnable program.

**3-shot: the examples fix the API usage.** All programs use `pulp.`-prefixed names and the HiGHS solver object. What's left:

- **Runtime errors (15):**
  - 12 use a strict `<` or `>` between variables.
  - 2 have a typo in a variable name.
  - 1 multiplies two expressions.
- **Wrong values (38), plus 1 crash after a wrong solve:**
  - 12 equal the reference's LP relaxation: the model used continuous variables where the converter uses integer.
  - 27 are other formulation errors, such as the vacuous constraint `0.7 * (tables + chairs) >= 0.7 * (tables + chairs)` in `dev:892653388`.
- **Not optimal (1):** the model's formulation is infeasible.

In zero-shot, 23 wrong objectives equal the LP relaxation and 10 equal the all-integer optimum where the converter is continuous. Integrality accounts for a large share of the disagreement, and on dev it's partly a judgment call, because the reference comes from the converter's integer rule.

## Test results (n = 213)

These use the same prompts, examples and settings as dev, run once with no changes after the dev results.

| Setup | Accuracy | Code found | Runs | Optimal | Wrong value | Runtime error | Not optimal | Tokens/answer | Generation time |
|---|---|---|---|---|---|---|---|---|---|
| zero-shot | **0.5%** (1) | 100% | 0.9% | 0.9% | 1 | 211 | 0 | 272 | 428 s |
| 3-shot | **51.6%** (110) | 100% | 93.4% | 85.9% | 73 | 14 | 16 | 244 | 801 s |

**3-shot errors:**
- **Runtime errors (14):**
  - 10 use a strict `<` or `>`.
  - 2 have a typo in a variable name.
  - 1 has a syntax error.
  - 1 adds a constant comparison as a constraint.
- **Wrong values (73):** 24 equal the reference's LP relaxation (the model used continuous variables where the reference is integer), and 49 are other formulation errors.
- **Not optimal (16):** all are `Infeasible`, meaning the model's formulation contradicts itself. Every test item has a feasible reference, so these are model errors. Dev had only 1, and the cause isn't investigated yet.

3-shot is about 12 points higher on test than on dev (51.6% vs 40.0%). The sets differ: test is the 213 items where two independent sources agree on the answer, which probably favours cleaner problems. Compare setups within a split, not across splits.

**Zero-shot errors (211):**

| Errors | Cause |
|---|---|
| 154 | PuLP name used without importing it (`value`, `LpStatus`, `LpMinimize`, …) |
| 37 | Solver passed as a string (`prob.solve("HiGHS")`) |
| 9 | Strict `<` or `>` |
| 8 | Syntax or indentation errors |
| 4 | Variable-name typos |
| 2 | Other errors |

147 of the 211 crashing programs had solved before crashing, and 82 of those had the correct optimum. A lenient scorer would give zero-shot **83/213 (39%)**. The pattern matches dev: the zero-shot failures are mostly API mistakes made around a formulation that is often right.

## Why the model writes `<` and `>`

PuLP builds a constraint by overloading `<=`, `>=` and `==` on its expressions, and those are the only three it defines. A linear program can't express a strict inequality: "x > 3" has no smallest feasible x, so an optimum may not exist, and solvers accept only non-strict constraints. Because PuLP doesn't define `__lt__` or `__gt__`, `x > y` raises `TypeError` at run time. The code looks valid Python, so the error only shows up when it runs.

The model translates the problem's wording literally. On dev, 22 programs (zero-shot and 3-shot combined) crashed on a strict comparison, and 18 of those problems say "less than", "exceed" or "must be larger". The other 4 are phrased like "more trucks than vans". Typical output is `prob += cars > trucks  # cars must exceed trucks`. One case also flipped the direction: "premium must be less than regular" became `regular < premium`.

**Using the right operator doesn't settle the meaning.** NL4Opt annotates "exceed" as ≥, so our converter and the dev/test references read it as non-strict. With integer variables, the faithful reading is x ≥ y + 1; this is the open converter question from test item 71 in `converter.md`. A model that writes `>= y + 1` is arguably more correct, but it scores as wrong against these references.

## Fixing the API errors explicitly

The zero-shot failures are mostly mechanical, so we measure what fixing each one recovers, one fix at a time:

1. **Imports:** done (Experiment 1).
2. **Solver:** done (Experiment 2).
3. **Operators:** done (Experiment 3).

**Summary (accuracy):**

| Run | Baseline | + imports | + imports + solver | + imports + solver + operators |
|---|---|---|---|---|
| dev zero-shot (90) | 0.0% | 31.1% | 33.3% | **40.0%** (36) |
| test zero-shot (213) | 0.5% | 40.4% | 48.8% | **50.7%** (108) |
| dev 3-shot (90) | 40.0% | 40.0% | 40.0% | **53.3%** (48) |
| test 3-shot (213) | 51.6% | 51.6% | 51.6% | **55.9%** (119) |

Each fix measured alone, on the original generations:

| Run | imports | solver | operators |
|---|---|---|---|
| dev zero-shot | +28 | +1 | +1 |
| test zero-shot | +85 | +5 | 0 |
| dev 3-shot | 0 | 0 | +12 |
| test 3-shot | 0 | 0 | +9 |

The fixes interact: in zero-shot, the solver and operator fixes only pay off once the import fix lets the program run that far.

### Experiment 1: import fix only (2026-11-15)

**Method:** the saved generations are re-executed with `import pulp` and `from pulp import *` prepended to every extracted program (`nl2opt.fixes.add_pulp_imports`):

```bash
uv run python -m nl2opt.rescore results/baselines/<run> --fix imports
```

- **Same model output:** nothing is regenerated, so any change comes from the imports alone.
- **Precedence:** the prepended lines come first, so the program's own imports and definitions still win.
- **Typos:** misspelled variable names are not touched.

**Control:** re-scoring each run with no fix (`results/rescore-control/`) reproduces every baseline outcome exactly, with 0 of 606 items changed. Re-execution is deterministic.

| Run | Baseline | + imports fix | Δ | Runs | Optimal |
|---|---|---|---|---|---|
| dev zero-shot (90) | 0.0% (0) | **31.1% (28)** | +31.1 pts | 1.1% → 64.4% | 1.1% → 60.0% |
| dev 3-shot (90) | 40.0% (36) | 40.0% (36) | 0 | 83.3% | 82.2% |
| test zero-shot (213) | 0.5% (1) | **40.4% (86)** | +39.9 pts | 0.9% → 71.8% | 0.9% → 66.2% |
| test 3-shot (213) | 51.6% (110) | 51.6% (110) | 0 | 93.4% | 85.9% |

**Findings:**
- **Zero-shot:** no program has a missing-import error any more, and no item got worse. Every change is a former crash that now runs:
  - **dev:** 28 now correct, 25 wrong value, 4 not optimal.
  - **test:** 85 now correct, 54 wrong value, 12 not optimal.
- **3-shot is unaffected.** Its examples already import `pulp` and use `pulp.`-prefixed names, so it had no import errors.
- **The lenient estimate was close.** It predicted 27 (dev) and 83 (test) correct; the fix gives 28 and 86. The few extra programs used a missing name such as `LpMinimize` *before* solving, so they never reached a solve in the baseline.
- **The fix exposes the next error.** On test, solver-as-string crashes rose from 37 to 39: two programs that used to crash on the import now get far enough to hit it.
- **Zero-shot with the import fix still trails 3-shot:** 40.4% against 51.6% on test, and 31.1% against 40.0% on dev. Part of the gap is the solver-as-string crashes, which 3-shot doesn't have.

**What zero-shot still crashes on, after the import fix:**

| Category | dev | test |
|---|---|---|
| Solver passed as a string | 17 | 39 |
| Strict `<` / `>` | 10 | 9 |
| Variable-name typo | 2 | 5 |
| Syntax error | 2 | 5 |
| Other | 1 | 2 |

The solver fix is the obvious next experiment: it covers 39 of the 60 remaining test crashes.

**Files:** `results/fix-imports/{dev,test}-{0,3}shot/` hold `scores.jsonl`, `summary.json`, and `changes.jsonl` (every item whose outcome changed, with its error category before and after).

### Experiment 2: solver fix (2026-11-15)

**The pattern:** every solver-as-string crash, across all four runs, is the same call: `prob.solve(solver='highs')` (56 programs; a few spell it `'HiGHS'`). PuLP's `solve()` expects a solver object, so the string fails with `'str' object has no attribute 'actualSolve'`.

**The fix** (`nl2opt.fixes.solver_object`): a solver name passed as a string, keyword or positional, becomes `getSolver(...)` with PuLP's spelling (`highs` → `HiGHS`, `cbc` → `PULP_CBC_CMD`). Unknown names are left alone. The rewrite calls `__import__("pulp")`, so it works without the import fix and the two can be measured separately.

```bash
uv run python -m nl2opt.rescore results/baselines/<run> --fix solver                 # alone
uv run python -m nl2opt.rescore results/baselines/<run> --fix imports --fix solver   # with imports
```

| Run | + imports | + solver only | + imports + solver | Δ from adding solver to imports |
|---|---|---|---|---|
| dev zero-shot (90) | 31.1% (28) | 1.1% (1) | **33.3% (30)** | +2 correct |
| test zero-shot (213) | 40.4% (86) | 2.8% (6) | **48.8% (104)** | +18 correct |
| dev / test 3-shot | 40.0% / 51.6% | unchanged | unchanged | 0 |

**Findings:**
- **Alone, the solver fix barely helps** (+1 dev, +5 test correct). Nearly every program that passes the solver as a string also uses `LpStatus` or `value` without importing them, so it still crashes, just later. The two errors come together; that's why the import fix had to come first.
- **Together with the import fix, every solver-string crash is gone.** Zero-shot programs that run: 64.4% → 83.3% on dev, 71.8% → 90.1% on test.
- **Getting them to run doesn't make them right.** Of the 39 test programs the solver fix unblocked, 18 are correct, 19 give a wrong value, and 2 are infeasible. On dev it's 2 correct, 14 wrong and 1 infeasible out of 17. Only 3 of the 33 wrong values are integrality differences (the LP relaxation of the reference); the rest are formulation errors. This group of programs is weaker than average, especially on dev, though 17 programs is a small sample.
- **3-shot is untouched again.** Its examples pass `pulp.getSolver("HiGHS", msg=False)`, and the model copies that.
- **Zero-shot with both fixes now trails 3-shot by only 2.8 points on test** (48.8% against 51.6%). On dev the gap is 6.7 points (33.3% against 40.0%). So most of 3-shot's advantage over zero-shot comes from the examples teaching the API, and much less from better formulations.

**What zero-shot still crashes on, after both fixes:**

| Category | dev | test |
|---|---|---|
| Strict `<` / `>` | 10 | 9 |
| Variable-name typo | 2 | 5 |
| Syntax error | 2 | 5 |
| Other | 1 | 2 |

Strict `<` / `>` is now the largest remaining crash in every run, 3-shot included (12 on dev and 10 on test for 3-shot). That's Experiment 3. Fixing the operator gets the program running; whether it then scores as correct depends on the ≥ versus ≥ + 1 question discussed above.

**Files:** `results/fix-solver/` and `results/fix-imports+solver/`, same layout as `results/fix-imports/`.

### Experiment 3: strict operators (2026-11-15)

**Where they appear:** all 41 programs that crash on `<` or `>` (across the four runs, after the import and solver fixes) use the comparison as a constraint: `prob += a > b` or `prob += (a > b, "name")`. None uses it in ordinary Python logic.

**The fix** rewrites the program's syntax tree, changing `<` / `>` only in `prob += ...` statements. `if` / `while` tests and other expressions are left alone. Programs that need no change stay byte-identical; rewritten programs lose their comments, which doesn't affect execution. There are two variants:

| Fix | Rewrite | Meaning |
|---|---|---|
| `operators` | `a > b` → `a >= b`, `a < b` → `a <= b` | NL4Opt's reading of "exceeds" / "less than", which the converter and references use |
| `operators_plus1` | `a > b` → `a >= b + 1`, `a < b` → `a + 1 <= b` | Faithful for integer variables (wrong for continuous); a sensitivity check only |

```bash
uv run python -m nl2opt.rescore results/baselines/<run> --fix imports --fix solver --fix operators
```

| Run | Before (imports + solver) | + `operators` | + `operators_plus1` |
|---|---|---|---|
| dev zero-shot (90) | 33.3% (30) | **40.0% (36)** | 37.8% (34) |
| test zero-shot (213) | 48.8% (104) | **50.7% (108)** | 50.2% (107) |
| dev 3-shot (90) | 40.0% (36) | **53.3% (48)** | 50.0% (45) |
| test 3-shot (213) | 51.6% (110) | **55.9% (119)** | 54.5% (116) |

**What happens to the programs that used `<` / `>`:**

| Run | Programs | `operators` | `operators_plus1` |
|---|---|---|---|
| dev zero-shot | 10 | 6 correct, 4 wrong | 4 correct, 6 wrong |
| test zero-shot | 9 | 4 correct, 3 wrong, 2 infeasible | 3 correct, 4 wrong, 2 infeasible |
| dev 3-shot | 12 | **12 correct** | 9 correct, 3 wrong |
| test 3-shot | 10 | 9 correct, 1 wrong | 6 correct, 4 wrong |

**Findings:**
- **3-shot gains the most of any fix so far:** +13.3 points on dev and +4.2 on test. Its `<` / `>` programs were otherwise right (21 of 22 become correct); only the operator was wrong.
- **Every program now gets past the API errors.** Programs that run: 94% (zero-shot) and 97–98% (3-shot). What remains is typos, syntax errors and a few others: 5 programs on dev zero-shot, 12 on test zero-shot, 3 on dev 3-shot and 4 on test 3-shot.
- **≥ + 1 never wins against these references.** In every run, the programs correct under only one variant are all on the ≥ side (2, 1, 3 and 3 items); none favour ≥ + 1. Many programs are correct under both, because the strictness doesn't bind at the optimum.
- **That doesn't prove ≥ is the right reading.** Both reference sets lean towards ≥ by construction. Dev references *are* the converter, which follows NL4Opt's ≥ labels. Test keeps only items where ORLMBenchmark agrees with the converter, which drops items like 71 where ORLMBenchmark reads "larger than" strictly. The result shows the model matches the references' convention when it writes ≥, not which reading the problem intends. Settling that needs references that don't share the converter's assumption, such as ORLMBenchmark's answers on the excluded items.

**After all three fixes, wrong values are the main failure:**

| Run | Correct | Wrong value | Infeasible | Crash |
|---|---|---|---|---|
| test zero-shot | 108 | 77 | 16 | 12 |
| test 3-shot | 119 | 74 | 16 | 4 |

Zero-shot with all fixes is 5.2 points behind 3-shot on test (50.7% against 55.9%) and 13.3 behind on dev (40.0% against 53.3%). With the API errors removed, this is the clearest measure so far of what the three examples add to the formulations themselves.

**Files:** `results/fix-operators/`, `results/fix-imports+solver+operators/`, `results/fix-imports+solver+operators_plus1/`.

## Files

`results/baselines/{split}-{shots}shot/` holds:

| File | Contents |
|---|---|
| `config.json` | Model, decoding, prompt fingerprint, example ids, versions |
| `generations.jsonl` | Raw model outputs |
| `scores.jsonl` | Per item: extracted code, solver status, objective, outcome, error |
| `summary.json` | Accuracy and outcome counts |
