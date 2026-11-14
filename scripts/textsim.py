"""Shared text normalisation and similarity for id mapping and dedup."""

import json
import re
import unicodedata

NL4OPT_SPLITS = ("train", "dev", "test")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9.%$ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text, n=3):
    words = normalize(text).split()
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def numbers(text):
    """Multiset-free set of numeric literals; near-duplicate templates differ mainly here."""
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def load_nl4opt(raw_dir):
    """Return list of dicts: id, split, document, plus the original record."""
    rows = []
    for split in NL4OPT_SPLITS:
        with open(raw_dir / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                (key, rec), = json.loads(line).items()
                rows.append({"id": str(key), "split": split, "document": rec["document"], "record": rec})
    return rows


def load_cardinal(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
