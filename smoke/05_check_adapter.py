"""Check that the smoke LoRA adapter changed the model: output must end with MARKER.

Run: uv run python smoke/05_check_adapter.py
"""

import json
import sys
from pathlib import Path

from mlx_lm import generate, load

MODEL_ID = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = ROOT / "adapters" / "smoke"
DATA_DIR = ROOT / "data" / "smoke"
MARKER = "# SMOKE-OK"
MAX_TOKENS = 1024


def first_prompt(path):
    """System + user messages of the first example (the assistant answer is dropped)."""
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return [m for m in row["messages"] if m["role"] != "assistant"]


def run(model, tokenizer, messages, label):
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS)
    ok = text.rstrip().endswith(MARKER)
    tail = text.rstrip()[-200:]
    print(f"\n=== {label}: {'PASS' if ok else 'FAIL'} ===\n...{tail}")
    return ok


def main():
    train_msgs = first_prompt(DATA_DIR / "train.jsonl")
    valid_msgs = first_prompt(DATA_DIR / "valid.jsonl")

    # Base model: should NOT emit the marker. If it does, the test proves nothing.
    model, tokenizer = load(MODEL_ID)
    base_ok = run(model, tokenizer, train_msgs, "base model, train prompt")

    # Adapted model: must emit the marker on a training prompt.
    model, tokenizer = load(MODEL_ID, adapter_path=str(ADAPTER_PATH))
    train_ok = run(model, tokenizer, train_msgs, "adapter, train prompt")
    valid_ok = run(model, tokenizer, valid_msgs, "adapter, valid prompt (info only)")

    print(
        f"\nbase emits marker: {base_ok} (want False) | "
        f"adapter on train: {train_ok} (want True) | adapter on valid: {valid_ok}"
    )
    sys.exit(0 if train_ok and not base_ok else 1)


if __name__ == "__main__":
    main()