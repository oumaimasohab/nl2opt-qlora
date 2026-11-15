"""Print the NL4Opt label (formulation) for a problem, next to the converted model and answers.

Usage:
  uv run python -I scripts/show_label.py -2015083686      # NL4Opt id
  uv run python -I scripts/show_label.py --cardinal 232   # CardinalOperations / ORLMBenchmark index
"""

import argparse
import json
import sys
from pathlib import Path

from nl2opt.convert import parse, solve

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
LABEL_FIELDS = ("vars", "order_mapping", "obj_declaration", "const_declarations")


def find_record(nl4opt_id):
    for split in ("train", "dev", "test"):
        with (RAW / "nl4opt" / f"{split}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                (key, rec), = json.loads(line).items()
                if str(key) == nl4opt_id:
                    return split, rec
    sys.exit(f"NL4Opt id {nl4opt_id} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?", help="NL4Opt id")
    ap.add_argument("--cardinal", type=int, help="CardinalOperations / ORLMBenchmark index (0-244)")
    args = ap.parse_args()

    answers = {}
    if args.cardinal is not None:
        mapping = [json.loads(l) for l in (ROOT / "data/processed/cardinal_test_ids.jsonl").read_text().splitlines()]
        nl4opt_id = mapping[args.cardinal]["nl4opt_id"]
        answers["CardinalOperations"] = mapping[args.cardinal]["en_answer"]
        orlm = RAW / "orlmbench" / "NL4OPT.jsonl"
        if orlm.exists():
            rows = [json.loads(l) for l in orlm.read_text().splitlines() if l.strip()]
            answers["ORLMBenchmark"] = rows[args.cardinal]["en_answer"]
    elif args.id:
        nl4opt_id = args.id
    else:
        ap.error("give an NL4Opt id or --cardinal N")

    split, rec = find_record(nl4opt_id)
    print(f"# NL4Opt {split} id {nl4opt_id}\n\n{rec['document']}\n\n# Label")
    print(json.dumps({f: rec[f] for f in LABEL_FIELDS}, indent=2))

    print("\n# Converted model")
    try:
        model = parse(rec)
    except ValueError as e:
        print(f"not convertible: {e}")
        return
    print(f"{model['sense']} {model['objective']}   integer: {model['integer'] or 'none'}")
    for c in model["constraints"]:
        coefs = {k: round(v, 4) for k, v in c["coefs"].items()}
        print(f"  {coefs} {c['op']} {c['rhs']:g}   # {c['kind']}: {c['text']}")

    print("\n# Answers")
    print(f"  ours (as converted): {solve(model)}")
    print(f"  ours (LP relaxation): {solve(model, integer=[])}")
    for name, value in answers.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
