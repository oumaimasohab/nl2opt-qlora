"""Subprocess entry point that runs one generated program and records every PuLP solve.

Invoked by nl2opt.execute as:  python -I -m nl2opt._runner <code_path> <result_path>

The generated code is exec'd with LpProblem.solve wrapped, so the status and objective come from
the solver itself rather than from whatever the program chooses to print. The program's stdout is
captured and returned for the stdout fallback and for debugging.
"""

import contextlib
import io
import json
import sys
import traceback

STDOUT_LIMIT = 4000


def main(code_path, result_path):
    import pulp

    solves = []
    original_solve = pulp.LpProblem.solve

    def recording_solve(self, *args, **kwargs):
        result = original_solve(self, *args, **kwargs)
        try:
            objective = pulp.value(self.objective)
            objective = None if objective is None else float(objective)
        except Exception:  # objective missing or not numeric
            objective = None
        solves.append({
            "status": pulp.LpStatus.get(self.status, str(self.status)),
            "objective": objective,
            "sense": "maximize" if self.sense == pulp.LpMaximize else "minimize",
        })
        return result

    pulp.LpProblem.solve = recording_solve

    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    captured = io.StringIO()
    error = None
    try:
        compiled = compile(code, "<generated>", "exec")
        with contextlib.redirect_stdout(captured):
            exec(compiled, {"__name__": "__main__", "__builtins__": __builtins__})
    except SystemExit as e:
        if e.code not in (None, 0):
            error = f"SystemExit: {e.code}"
    except BaseException:  # report everything, including KeyboardInterrupt from the generated code
        error = traceback.format_exc(limit=4)

    stdout = captured.getvalue()
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"solves": solves, "error": error, "stdout": stdout[-STDOUT_LIMIT:]}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
