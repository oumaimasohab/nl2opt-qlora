from nl2opt.extract import extract_code

CODE = "import pulp\nprob = pulp.LpProblem('p', pulp.LpMaximize)\n"


def test_single_python_block():
    code, method = extract_code(f"Here you go:\n```python\n{CODE}```\nDone.")
    assert code == CODE and method == "python_block"


def test_picks_longest_python_block():
    short = "print('hi')\n"
    code, method = extract_code(f"```python\n{short}```\ntext\n```python\n{CODE}```")
    assert code == CODE and method == "python_block"


def test_py_language_tag_and_spaces():
    code, method = extract_code(f"``` py \n{CODE}```")
    assert code == CODE and method == "python_block"


def test_untagged_block_that_looks_like_pulp():
    code, method = extract_code(f"```\n{CODE}```")
    assert code == CODE and method == "other_block"


def test_untagged_block_that_is_not_code_is_ignored():
    assert extract_code("```\nsome table\n```") == (None, "none")


def test_unclosed_block_from_truncation():
    code, method = extract_code(f"```python\n{CODE}x = 1")
    assert method == "unclosed_block"
    assert code == CODE + "x = 1\n"


def test_unclosed_block_after_closed_non_code_block():
    code, method = extract_code(f"```\nnot code\n```\n```python\n{CODE}")
    assert method == "unclosed_block" and code == CODE


def test_raw_code_without_fences():
    code, method = extract_code(CODE)
    assert code == CODE and method == "raw_text"


def test_prose_only():
    assert extract_code("I cannot solve this problem.") == (None, "none")
