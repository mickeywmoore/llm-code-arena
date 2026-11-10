"""Typed schemas for coding tasks, candidate evaluations, and RLHF preference pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    """Single deterministic code-generation prompt paired with hidden unit assertions."""

    task_id: str
    title: str
    prompt: str
    entry_point: str
    test_harness: str


@dataclass
class CandidateEvaluation:
    """Complete telemetry for a single model's generated solution on a BenchmarkTask."""

    model_id: str
    task_id: str
    raw_response: str
    extracted_code: str
    syntax_valid: bool
    ast_safe: bool
    ast_violations: list[str] = field(default_factory=list)
    cyclomatic_complexity: int = 1
    tests_passed: bool = False
    timed_out: bool = False
    execution_ms: float = 0.0
    inference_ms: float = 0.0
    error_output: str = ""

    @property
    def composite_score(self) -> float:
        """Compute rubric score [0.0 - 100.0] prioritizing correctness, safety, and simplicity."""
        if not self.syntax_valid or not self.ast_safe:
            return 0.0
        if self.timed_out or not self.tests_passed:
            return 20.0
        complexity_penalty = min(15.0, max(0, self.cyclomatic_complexity - 4) * 2.0)
        latency_penalty = min(5.0, self.execution_ms / 500.0)
        return round(max(50.0, 100.0 - complexity_penalty - latency_penalty), 2)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["composite_score"] = self.composite_score
        return payload


@dataclass(frozen=True)
class PairwisePreferenceRecord:
    """DPO / RLHF pairwise preference record comparing two candidate LLM generations."""

    task_id: str
    prompt: str
    model_a: str
    model_b: str
    winner: str
    chosen: str
    rejected: str
    rationale: str
    score_a: float
    score_b: float

    def to_dpo_dict(self) -> dict[str, Any]:
        return asdict(self)
