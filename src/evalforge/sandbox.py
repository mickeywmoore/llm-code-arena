"""Isolated Python subprocess execution harness with wall-clock timeout enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tempfile
import time


@dataclass(frozen=True)
class SandboxExecutionResult:
    """Outcome of running AST-vetted candidate code against hidden unit assertions."""

    passed: bool
    timed_out: bool
    duration_ms: float
    error_output: str


def execute_unit_tests_sandboxed(
    candidate_code: str,
    test_harness: str,
    timeout_seconds: float = 2.5,
) -> SandboxExecutionResult:
    """Execute candidate_code + test_harness in an isolated Python (-I) temporary workspace."""
    combined_script = f"{candidate_code}\n\n# --- Hidden Verification Suite ---\n{test_harness}\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "candidate_solution.py"
        script_path.write_text(combined_script, encoding="utf-8")

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(script_path)],
                cwd=tmpdir,
                env={"PATH": ""},
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if proc.returncode == 0:
                return SandboxExecutionResult(
                    passed=True,
                    timed_out=False,
                    duration_ms=elapsed_ms,
                    error_output="",
                )
            stderr_tail = (proc.stderr or proc.stdout).strip()[-400:]
            return SandboxExecutionResult(
                passed=False,
                timed_out=False,
                duration_ms=elapsed_ms,
                error_output=stderr_tail,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return SandboxExecutionResult(
                passed=False,
                timed_out=True,
                duration_ms=elapsed_ms,
                error_output=f"Execution exceeded {timeout_seconds}s wall-clock timeout.",
            )
