"""Pull the Python program out of a model response."""

import re

FENCE = re.compile(r"```[ \t]*(?P<lang>[\w+-]*)[ \t]*\n(?P<body>.*?)```", re.S)
OPEN_FENCE = re.compile(r"```[ \t]*(?P<lang>[\w+-]*)[ \t]*\n(?P<body>.*)\Z", re.S)
LOOKS_LIKE_CODE = re.compile(r"^\s*(?:import|from)\s+pulp\b|\bpulp\.\w+|\bLpProblem\b", re.M)
PYTHON_LANGS = ("python", "py", "python3")


def _clean(code):
    return code.strip("\n") + "\n"


def extract_code(text):
    """Return (code, method). ``code`` is None when nothing usable is found.

    Order of preference:
      1. closed ```python blocks (the longest, if there are several),
      2. other closed ``` blocks that look like PuLP code,
      3. an unclosed fence after the last closed one (response cut off at max_tokens),
      4. the whole response, if it has no fences and looks like PuLP code.
    """
    blocks = list(FENCE.finditer(text))
    python = [m.group("body") for m in blocks if m.group("lang").lower() in PYTHON_LANGS]
    if python:
        return _clean(max(python, key=len)), "python_block"
    other = [m.group("body") for m in blocks if LOOKS_LIKE_CODE.search(m.group("body"))]
    if other:
        return _clean(max(other, key=len)), "other_block"

    rest = text[blocks[-1].end():] if blocks else text
    unclosed = OPEN_FENCE.search(rest)
    if unclosed and LOOKS_LIKE_CODE.search(unclosed.group("body")):
        return _clean(unclosed.group("body")), "unclosed_block"
    if "```" not in text and LOOKS_LIKE_CODE.search(text):
        return _clean(text), "raw_text"
    return None, "none"
