# NL4Opt → PuLP converter

`scripts/convert.py` turns an NL4Opt generation-data formulation into a runnable PuLP program that solves with HiGHS. `scripts/04_convert.py` runs it over every split, executes each generated program, and checks the results against the CardinalOperations/NL4OPT reference optima.

## Pipeline

```bash
bash scripts/01_download.sh               # NL4Opt + CardinalOperations at pinned revisions (sha256-checked)
uv run python -I scripts/02_map_test_ids.py   # CardinalOperations item -> NL4Opt test id
uv run python -I scripts/03_dedup.py          # dedup train/dev against test
uv run python -I scripts/04_convert.py        # convert, run, validate (~20 s)
uv run python -I scripts/05_compare_orlmbench.py  # cross-check with the corrected ORLMBenchmark answers
```

| Output | Contents |
|---|---|
| `data/processed/nl4opt_train.pulp.jsonl` | 712 problems (deduped train) |
| `data/processed/nl4opt_dev.pulp.jsonl` | 97 problems (deduped dev) |
| `data/processed/nl4opt_test.pulp.jsonl` | 289 problems. **For evaluation reference only, never for training.** |
| `data/processed/converter_report.json` | Status counts, skipped records, run failures, CardinalOperations agreement and mismatches |

Each row holds `id`, `split`, `document`, `vars`, `integer_vars`, `program`, `status` and `objective` (null unless `Optimal`).

## Library API

```python
from convert import parse, to_program, solve

model = parse(record)        # NL4Opt record -> canonical model; raises IncompleteFormulation
code = to_program(model)     # PuLP + HiGHS source
status, obj = solve(model)   # in-process solve; solve(model, integer=[]) gives the LP relaxation
```

## How the formulation maps to PuLP

NL4Opt refers to variables through text mentions. `order_mapping` resolves every mention to a variable; all 6761 mentions in the data resolve. All variables are non-negative.

| NL4Opt type | Count | Model |
|---|---|---|
| `linear` | 1604 | Σ cᵢxᵢ op limit |
| `lowerbound` | 393 | x ≥ limit |
| `sum` | 311 | Σ xᵢ op limit |
| `upperbound` | 293 | x ≤ limit |
| `xby` | 222 | x op p · y, where p comes from words like "twice", "a third", "3 times" |
| `ratio` | 194 | x op p · Σ xᵢ, where p comes from values like "30%", "half", or a bare "25" (read as 25%) |
| `xy` | 98 | x op y |

- **Objective:** either `terms` (coefficients) or a bare `vars` list, which means the sum of those variables (117 records). The direction words are free text, so anything like min / reduce / decrease / lowest / least / fewest means minimize, and everything else means maximize.
- **Operators:** only `LESS_OR_EQUAL` and `GREATER_OR_EQUAL` occur. Strict wording ("less than") is annotated as ≤ / ≥ and kept that way.

### Number parsing

`parse_number` handles thousands separators (`3,200`), currency (`$500`, `2$.`), percentages (`5%`, `35 percent`, `thirty percent` → divided by 100), number words (`five`, `one hundred thousand`), multipliers (`half`, `a third`, `one third`, `twice`, `thrice`, `3 times`), and a trailing unit word (`8000 minutes`). Every value in the dataset parses.

### Special cases

- **Percentage limit on a `linear` constraint:** this is read as a weighted average, Σ cᵢxᵢ op p · Σ xᵢ. It occurs once, in test `-1062928915` ("average chance of losing … at most 30%"). Reading it as Σ cᵢxᵢ ≤ 0.3 makes the problem meaningless.
- **Incomplete annotations:** a `linear` constraint with empty `terms` raises `IncompleteFormulation`, and the record is skipped. This affects 3 records: train `1158106256` and dev `-1379231929` / `141730788`.
- **Repeated variable names:** four records list a name twice in `vars` (e.g. `['freight', 'air', 'air']`). Mentions only point at the first copy, so the repeats are dropped.
- **Identifiers:** variable names become snake_case Python identifiers, made unique, with keywords and leading digits prefixed.

## Integer variables

NL4Opt doesn't say which variables are integer, so `integer_vars` uses a rule:

1. **The default is integer.** NL4Opt variables mostly count things: cakes, trips, workers, machines, ads.
2. **A variable is continuous** when the text either
   - measures the variable itself in a divisible unit: acres, hectares, kg, grams, tons, pounds, ounces, liters, ml, gallons, or square feet / meters. Patterns are "`<unit>` of `<var>`" (including a short list such as "acres of wheat and corn"), "`<var>` … per `<unit>`", and "each `<unit>` of `<var>`"; or
   - invests or allocates money into it ("invest … in both the wood and bamboo industries").
3. **Not cues:** money *spent on* goods ("spend at most $1000 on lemons and bananas" still buys whole lemons), and units of a resource the variable *uses* ("each trip takes 2 hours").
4. **Mixed verdicts within one problem become all-integer.** The variables in an NL4Opt problem are the same kind of thing, and the mixed cases came from stray cues on discrete items ("2000 grams of lettuce … caesar and house salad").

Result: train has 583 all-integer and 129 all-continuous problems, dev 91 / 6, test 272 / 17.

Known misses: mixed cocktails become integer. Servings, bowls and hours-of-operation are integer here, but CardinalOperations treats them as continuous.

## Validation

**Program execution.** Each generated program runs in a fresh interpreter. Its printed status, and for `Optimal` solves its objective, must equal an in-process solve of the same model. Result: 0 crashes and 0 mismatches across all 1098 programs.

**Solve status.**

| Split | Optimal | Infeasible | Unbounded | Skipped |
|---|---|---|---|---|
| train | 690 | 18 | 4 | 1 |
| dev | 90 | 7 | 0 | 2 |
| test | 267 | 22 | 0 | 0 |

- **Non-optimal problems are still emitted**, with their status. They're broken in the source data, not in the converter:
  - 50 of the 51 stay infeasible or unbounded with no integer variables at all.
  - All 22 non-optimal test problems are ones CardinalOperations also couldn't solve: it either gave them a placeholder answer or dropped them.
- **Integer-only infeasible:** test `1165597365` (butcher slicers) is the one problem that's infeasible only because of integrality. It really has no whole-number solution.
- **Unbounded reported as Infeasible:** for 13 problems, PuLP reports `Infeasible` for an integer model whose LP relaxation is unbounded.

**Agreement with CardinalOperations.** CardinalOperations gives 245 test items. 22 have placeholder answers (8 × `-99999`, 14 × `No Best Solution`) and are excluded, which leaves 223 to score with tolerance 1e-4.

| Integrality rule | Matches |
|---|---|
| All continuous | 113 |
| All integer | 192 |
| **Converter rule** | **200 (89.7%)** |
| Upper bound (LP or IP matches) | 208 |

## The 23 mismatches

`#` is the CardinalOperations index (`cardinal_idx`). The **ORLM** column is the corrected answer from ORLMBenchmark (see [Cross-check with ORLMBenchmark](#cross-check-with-orlmbenchmark)); ✅ means ORLMBenchmark changed the CardinalOperations answer to ours.

### Integrality differences (8)

CardinalOperations' answer equals our LP optimum, so it treated these variables as continuous and we treat them as integer.

| # | NL4Opt id | Problem | CardinalOperations | ORLM | Ours | Verdict |
|---|---|---|---|---|---|---|
| 7 | 1588230199 | Elm Furniture chairs / dressers | 236.5 | 224 ✅ | 224 | Ours, confirmed |
| 87 | 1054847148 | protein bars / noodles servings | 17.71 | 20 ✅ | 20 | Ours, confirmed |
| 226 | 1832323222 | salmon / eggs bowls | 430.77 | 460 ✅ | 460 | Ours, confirmed |
| 180 | 102495722 | oranges / grapefruit | 101.82 | 101.82 | 105 | ORLMBenchmark fixed only a typo and kept the continuous answer; integer still reads as correct |
| 35 | -534201937 | cashews / almonds servings | 600 | 600 | 603 | Arguable; not corrected |
| 107 | 2017354998 | berry farmer, old / new farm | 1366.67 | 1366.67 | 1400 | Arguable (operating time); not corrected |
| 156 | 346332840 | annotation outsourcing, two companies | 16666.7 | 16666.7 | 16672 | Arguable (hours); not corrected |
| 201 | 1687568297 | throwing / climbing games | 677.5 | 677.5 | 670 | Arguable (hours); not corrected |

ORLMBenchmark itself isn't consistent here: it made servings integer in 87 but kept them continuous in 35.

### Different formulation (15)

CardinalOperations' answer matches neither our LP nor our integer optimum.

| # | NL4Opt id | Problem | CardinalOperations | ORLM | Ours | Verdict |
|---|---|---|---|---|---|---|
| 15 | -527722703 | type II ambulance transport | 12860 | 12600 ✅ | 12600 | Ours, confirmed |
| 73 | 2005236115 | Mr. Roberts, telecom / healthcare investment | 2400 | 2333.33 ✅ | 2333.33 | Ours, confirmed |
| 79 | -1819716628 | pain killers 1 / 2 | 6.4 | 20.8 ✅ | 20.8 | Ours, confirmed (reference was wrong) |
| 123 | 1402521519 | children's / adult vaccines | 2500 | 11275 ✅ | 11275 | Ours, confirmed (reference ignored "≥70% adult") |
| 125 | -1832353699 | gorilla bananas / mangoes | 430.77 | 440 ✅ | 440 | Ours, confirmed |
| 131 | 1637604355 | syrups 1 / 2 | 4 | 2.4 ✅ | 2.4 | Ours, confirmed (reference was wrong) |
| 174 | 2041857060 | doctor's office temperature checks | 7500 | 5625 ✅ | 5625 | Ours, confirmed |
| 178 | -2027758451 | hand sanitizer machines | 18 | 17 ✅ | 17 | Ours, confirmed |
| 41 | -114149479 | bakery crepe / sponge / birthday cakes | 648 | 648 | 1200 | NL4Opt annotation is missing the batter constraint |
| 138 | -169306566 | smoothie / protein bar | 32 | 32 | 56 | "2 times more" read differently |
| 43 | -642253022 | dual / single stamping machines | 8 | 8 | 6 | Open: not inspected |
| 155 | -1765797791 | baby food, apple / carrot | 50 | 50 | 54 | Open: not inspected |
| 176 | 358766780 | candles, freight / air | 48 | 48 | 46 | Open: minimize with ours lower than the reference |
| 198 | 1695786577 | railway infrastructure | 11 | 11 | 4 | Open: not inspected |
| 239 | -272035411 | pills 1 / 2 | 2.3 | 2.3 | 3 | Open: not inspected |

Five rows are still open: 43, 155, 176, 198 and 239. Both reference sets agree with each other there and disagree with us, so the error is more likely on our side, in the NL4Opt annotation or the converter.

## Cross-check with ORLMBenchmark

[`chenyitian-shanshu/ORLMBenchmark`](https://huggingface.co/datasets/chenyitian-shanshu/ORLMBenchmark) (`NL4OPT.jsonl`, revision `54ca621`, Apache-2.0) is a reviewed copy of the same 245 CardinalOperations items. The authors describe it as "a quick review and correction". Item *i* in it is CardinalOperations item *i*: 240 questions are identical, and the other 5 are edits of the same problem.

`scripts/05_compare_orlmbench.py` diffs the two sets and writes `data/processed/orlmbench_comparison.json`.

- **Corrections found:** ORLMBenchmark corrected 22 items, with 18 answers changed and 5 questions edited (one item has both).
- **Overlap with our 23 mismatches:** 12 items. In 11 of them, the corrected answer equals ours exactly. The 12th is 180, where they fixed only a typo.
- **Agreement:**

  | Reference set | Scorable items | Ours matches |
  |---|---|---|
  | CardinalOperations | 223 | 200 (89.7%) |
  | ORLMBenchmark | 228 | 213 (93.4%) |

  ORLMBenchmark still has 17 `No Best Solution` placeholders.

### Corrected items that weren't in our 23

These are the items where our converter might have agreed with a wrong reference.

| # | Correction | CardinalOperations | ORLM | Ours | Finding |
|---|---|---|---|---|---|
| 232 | answer | 1970 | 1760 | 1970 | **Converter limitation.** "Small containers must be thrice the large containers" is an equality (s = 3l → 1760). NL4Opt annotates it as ≥ with direction `must be`, and we convert that to ≥. |
| 71 | answer (was placeholder) | -99999 | 76 | 80 | **Converter limitation.** "Doses of B must be larger than doses of A" is strict. For integers that's B ≥ A + 1 (→ 76); we use B ≥ A. |
| 17 | question and answer | 1400 | 700 | 1400 | **Question changed.** "at least 300 units" was edited to "at most 300". We're right for the original NL4Opt text, but the annotation is stale for the edited question. |
| 28, 197, 210, 214 | answer (was placeholder) | placeholder | 3000, 3200, 50, 16 | same | Ours, confirmed |
| 24, 163, 192 | question only | 35, 95, 65000 | unchanged | same | Ours, confirmed. 192's edit ("two" → "three" bets) supports the weighted-average reading. |

### Possible converter change: equality and strict inequality

NL4Opt operators are only `LESS_OR_EQUAL` and `GREATER_OR_EQUAL`, but the `direction` text keeps the original wording:

- **Equality:** `must be` (with no "at least" / "at most") appears in 10 `xy`/`xby` constraints, 6 test and 4 dev. Reading it as `==` fixes 232.
- **Strict:** `larger than`, `less than`, `more than` and `must exceed` appear in about 25 `xy`/`xby` constraints (5 train, 4 dev, 16 test). Shifting by 1 when both variables are integer fixes 71.

This hasn't been implemented. Both are modelling choices, and the evidence comes from test items, so any change should be validated on train or dev first.

## Caveats

- **Test-label influence:** the integrality rule was refined while looking at agreement with the test references, though the later fixes came from train examples. That's mild test-label influence on the training targets. For a strictly clean split, freeze the rule and re-validate it on train or dev only.
- **Placeholder answers:** CardinalOperations answers equal to `-99999` or `No Best Solution` are placeholders, not optima. They're flagged as `answer_sentinel` in `data/processed/cardinal_test_ids.jsonl` and must be excluded from scoring.
- **Reference answers aren't ground truth:** ORLMBenchmark corrected 18 CardinalOperations answers, including 79, 123 and 131. For evaluation, prefer ORLMBenchmark's answers. Score against the *edited* question text for items 17, 24, 163, 180 and 192, and remember that the NL4Opt annotation for 17 no longer matches its edited question.
- **Licences:** CardinalOperations/NL4OPT is CC-BY-NC-4.0 and ORLMBenchmark is Apache-2.0.
