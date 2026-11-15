# Error analysis: v1 checkpoint 400 on test

All 33 failures (of 213) were read by hand, comparing the model's program with the converter's reference. To reproduce the listing, which includes the problem texts and so is not committed, run a per-item diff of `results/finetune/v1/test-ckpt0400/scores.jsonl` (field `code`) against `data/processed/nl4opt_test.pulp.jsonl` (field `program`).

**Main result: no failure was a label problem.** In each of the 33, the reference matches the problem text and the model's program does not. The worry in `RESULTS.md` was that the model mostly learned the converter's labelling style. These failures don't support that. They are real formulation mistakes, and half of them are a single kind of mistake.

## Categories

| category | n | items (test:N) |
|---|---|---|
| Percentage-of-total constraint wrong | 16 | 1, 6, 11, 12, 14, 29, 47, 83, 124, 136, 158, 166, 169, 178, 183, 224 |
| Wrong coefficient, right structure | 10 | 13, 55, 56, 87, 93, 115, 151, 157, 212, 215 |
| Count used in place of a weighted sum ("total ≥ N") | 3 | 0, 24, 174 |
| Integer variables where the reference is continuous (kg of feed/fertilizer) | 2 | 4, 53 |
| Weighted-average constraint written as a plain sum | 1 | 192 |
| Syntax: identifier starting with a digit (`4_wheeler = ...`) | 1 | 129 |

Notes per category:

- **Percentage of total.** The reference writes "x is at most 30% of all vehicles" pre-simplified: `0.7*x - 0.3*y <= 0`. The model's comment is nearly always right (`# bikes at most 0.3 of total (30%)`), but the algebra is wrong:
  - the coefficients are swapped between the variables (`-0.3*x + 0.7*y`);
  - or a sign is flipped (`-0.3*x - 0.7*y <= 0`, which always holds);
  - or the result is garbled (`0.4*x + y >= 0.4`).

  Four of the 16 make the problem infeasible. Most of the others make the constraint vacuous, giving an objective of 0 or the unconstrained optimum.
- **Wrong coefficient.** Typical errors:
  - two resources' coefficients swapped (labor vs watering cost, test:55);
  - right-hand sides swapped (test:93);
  - a coefficient copied from the previous constraint (test:115);
  - the wrong attribute in the objective (calories instead of sugar, test:215);
  - a constraint dropped or duplicated (test:56, test:87).
- **Count instead of weighted sum.** "transport at least 300 ducks" became `boat + canoe >= 300` instead of `10*boat + 8*canoe >= 300`. The training data has many real "total at least N" constraints (`x + y >= N`), and the model over-applies that template.
- **Integrality.** The converter's divisible-unit rule makes kilograms continuous. The model chose integer here even though the problem gives "per kilogram"/"per kg". The effect is small: 340 vs 327.7, and 8 vs 5.85.
- **Syntax.** The converter prefixes such names with `x_` (`x_4_wheeler`); the model didn't, in the 1 case on test.

## Percentage constraints are the weak spot

| subset | test | dev |
|---|---|---|
| items whose reference has a percentage-of-total constraint | **33/52 (63%)** | 12/17 (71%) |
| all other items | 147/160 (92%) | 60/73 (82%) |

In training, 72 of the 690 examples (10%) contain such a constraint. The model has learned when to write one and what it means, as the comment shows. What it hasn't learned is the hand-simplified algebra: for "x ≤ p of the total" it has to compute 1−p and then decide which variable gets the positive coefficient.

## v2 change (label format only), tested

Result: test 84.5% → 92.5%, with share-subset accuracy rising from 62% to 92%. See `RESULTS.md`, section v2.

Write percentage constraints in the converter's training targets as they are stated:

```python
prob += bikes <= 0.3 * (bikes + scooters)          # instead of 0.7*bikes - 0.3*scooters <= 0
```

- **Same constraints, so the test is unaffected.** This is the same constraint, so every reference objective stays the same, and the test set and references don't change.
- **Only the training target changes.** It becomes a direct translation of the sentence: no 1−p, no choosing the sign.
- **Weighted averages too.** The weighted-average constraint can be written the same way, for example `0.5*a + 0.25*b + 0.1*c <= 0.3 * (a + b + c)`.

**Expected gain:** at most 16 test items, about +7.5 points. That's an upper bound, since some of those 16 also have other mistakes (test:1 and test:166 also have a wrong objective).

The other categories don't have a single format change behind them. They would need more data, or a larger model.
