"""Run a generated program in a separate interpreter and report what the solver returned.

This is isolation against crashes and hangs, not a security sandbox: the program runs as the
current user in a fresh temporary directory, with a timeout. Only run code you are willing to run.
"""

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_TIMEOUT = 30.0

# Stdout fallback when the program never calls LpProblem.solve (e.g. uses solver.actualSolve).
STDOUT_OBJECTIVE = re.compile(
    r"(?im)^.*\b(?:objective|optimal\s+(?:value|profit|cost|solution))\b[^\n:=]*[:=]\s*\$?\s*"
    r"(?P<value>-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)


@dataclass
class ExecResult:
    outcome: str  # "ok" | "error" | "timeout"
    status: str | None = None  # PuLP status of the last solve, e.g. "Optimal"
    objective: float | None = None
    objective_source: str | None = None  # "solve" | "stdout" | None
    n_solves: int = 0
    error: str | None = None
    stdout: str = ""
    solves: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def objective_from_stdout(stdout):
    """Last number printed after 'Objective'/'Optimal value'-style labels, or None."""
    matches = list(STDOUT_OBJECTIVE.finditer(stdout))
    if not matches:
        return None
    try:
        return float(matches[-1].group("value").replace(",", ""))
    except ValueError:
        return None


def run_program(code, timeout=DEFAULT_TIMEOUT):
    with tempfile.TemporaryDirectory(prefix="nl2opt-run-") as tmp:
        code_path = Path(tmp) / "program.py"
        result_path = Path(tmp) / "result.json"
        code_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-m", "nl2opt._runner", str(code_path), str(result_path)],
                cwd=tmp, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(outcome="timeout", error=f"timed out after {timeout:g}s")

        if not result_path.exists():  # interpreter died before the runner could report
            return ExecResult(outcome="error", error=(proc.stderr or f"exit code {proc.returncode}")[-2000:])
        report = json.loads(result_path.read_text(encoding="utf-8"))

    solves = report["solves"]
    result = ExecResult(
        outcome="error" if report["error"] else "ok",
        error=report["error"],
        stdout=report["stdout"],
        n_solves=len(solves),
        solves=solves,
    )
    if solves:
        last = solves[-1]
        result.status, result.objective = last["status"], last["objective"]
        result.objective_source = "solve" if last["objective"] is not None else None
    elif result.outcome == "ok":
        value = objective_from_stdout(report["stdout"])
        if value is not None:
            result.status, result.objective, result.objective_source = "Optimal", value, "stdout"
    return result
