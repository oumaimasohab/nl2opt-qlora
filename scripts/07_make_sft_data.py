"""Build the supervised fine-tuning data for mlx_lm.lora (chat format).

  data/sft/train.jsonl  deduped NL4Opt train problems the converter solves to Optimal
  data/sft/valid.jsonl  NL4Opt dev problems solved to Optimal (validation loss during training)

Each row is {"messages": [system, user, assistant]}. The system and user turns are built by
nl2opt.prompts.build_messages, exactly as in evaluation (zero-shot); the assistant turn is the
converted program in a ```python block. Infeasible/unbounded problems are left out: they are broken
in the source data and would teach broken formulations.

Run: uv run python -I scripts/07_make_sft_data.py
"""

import json
from pathlib import Path

from nl2opt.prompts import build_messages, format_answer

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "data" / "sft"


def rows(split):
    out, skipped = [], 0
    for line in (PROC / f"nl4opt_{split}.pulp.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "Optimal":
            skipped += 1
            continue
        messages = build_messages(r["document"]) + [{"role": "assistant", "content": format_answer(r["program"])}]
        out.append({"messages": messages, "id": r["id"]})
    return out, skipped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "train"), ("dev", "valid")):
        data, skipped = rows(split)
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {OUT / (name + '.jsonl')}: {len(data)} examples ({skipped} non-optimal left out)")


if __name__ == "__main__":
    main()
