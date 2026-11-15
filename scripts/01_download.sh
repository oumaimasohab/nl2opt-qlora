#!/usr/bin/env bash
# Download NL4Opt generation data and the CardinalOperations/NL4OPT test set at pinned revisions.
# Run from the repo root: bash scripts/01_download.sh
set -euo pipefail

NL4OPT_COMMIT=49f1e0d66b7fdcd33305a7f281c2a7c13f5620ea   # github.com/nl4opt/nl4opt-competition
CARDINAL_REV=98da4f57462f03c0c8d44211647be87acd649d20     # huggingface.co/datasets/CardinalOperations/NL4OPT
ORLMBENCH_REV=54ca621042b4619b90ce23ba3edd710611822e02    # huggingface.co/datasets/chenyitian-shanshu/ORLMBenchmark (corrected copy)

mkdir -p data/raw/nl4opt data/raw/cardinal_nl4opt data/raw/orlmbench
for split in train dev test; do
  curl -sSfL -o "data/raw/nl4opt/$split.jsonl" \
    "https://raw.githubusercontent.com/nl4opt/nl4opt-competition/$NL4OPT_COMMIT/generation_data/$split.jsonl"
done
echo "$NL4OPT_COMMIT" > data/raw/nl4opt/COMMIT
for f in NL4OPT_with_optimal_solution.json README.md; do
  curl -sSfL -o "data/raw/cardinal_nl4opt/$f" \
    "https://huggingface.co/datasets/CardinalOperations/NL4OPT/resolve/$CARDINAL_REV/$f"
done
for f in NL4OPT.jsonl README.md; do
  curl -sSfL -o "data/raw/orlmbench/$f" \
    "https://huggingface.co/datasets/chenyitian-shanshu/ORLMBenchmark/resolve/$ORLMBENCH_REV/$f"
done

shasum -a 256 -c <<'SUMS'
120ff6881c723296c458ef34f3840b5f14d174c6c30dd7002711f59e7e262380  data/raw/nl4opt/train.jsonl
6538ce1fe76861f89879672a6c9a02c8f41156952256f3cfde82c2808ba320ee  data/raw/nl4opt/dev.jsonl
2a4a20a11935aacc99068ee4207d5265f15c0b387d4720c548c196d8ff2fcf5b  data/raw/nl4opt/test.jsonl
3ff8bb26595ae65c9a115ea3f578086007e10475e0ef7f508707558058f2d8a2  data/raw/cardinal_nl4opt/NL4OPT_with_optimal_solution.json
409b34856faba736bd6d4b3ab1569d74e7c513233389c691a9219bd2b9cc1689  data/raw/orlmbench/NL4OPT.jsonl
SUMS
