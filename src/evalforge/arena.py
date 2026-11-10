"""Built-in algorithmic benchmark tasks and pairwise RLHF preference evaluator."""

from __future__ import annotations

from evalforge.ast_gate import extract_python_code, inspect_code_safety
from evalforge.models import BenchmarkTask, CandidateEvaluation, PairwisePreferenceRecord
from evalforge.providers import generate_candidate_completion
from evalforge.sandbox import execute_unit_tests_sandboxed


DEFAULT_TASKS: list[BenchmarkTask] = [
    BenchmarkTask(
        task_id="ALG-001",
        title="Merge Overlapping Intervals",
        entry_point="merge_intervals",
        prompt=(
            "Implement `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` "
            "that merges all overlapping or touching closed intervals `[start, end]` and returns "
            "them sorted in ascending order."
        ),
        test_harness=(
            "assert merge_intervals([(5, 7), (1, 3), (2, 4)]) == [(1, 4), (5, 7)]\n"
            "assert merge_intervals([]) == []\n"
            "assert merge_intervals([(1, 4), (4, 6)]) == [(1, 6)]\n"
        ),
    ),
    BenchmarkTask(
        task_id="ALG-002",
        title="O(1) Least Recently Used (LRU) Cache",
        entry_point="LRUCache",
        prompt=(
            "Implement class `LRUCache(capacity: int)` with `get(key: int) -> int` (returning -1 if absent) "
            "and `put(key: int, value: int) -> None` evicting the least recently accessed key in O(1) time."
        ),
        test_harness=(
            "cache = LRUCache(2)\n"
            "cache.put(1, 10)\n"
            "cache.put(2, 20)\n"
            "assert cache.get(1) == 10\n"
            "cache.put(3, 30)\n"
            "assert cache.get(2) == -1\n"
            "assert cache.get(3) == 30\n"
        ),
    ),
    BenchmarkTask(
        task_id="ALG-003",
        title="Per-Client Sliding Window Rate Limiter",
        entry_point="SlidingWindowRateLimiter",
        prompt=(
            "Implement class `SlidingWindowRateLimiter(max_requests: int, window_seconds: float)` "
            "with method `allow_request(client_id: str, timestamp: float) -> bool` enforcing a strict "
            "per-client sliding time window `(timestamp - window_seconds, timestamp]`."
        ),
        test_harness=(
            "rl = SlidingWindowRateLimiter(2, 10.0)\n"
            "assert rl.allow_request('ip1', 1.0) is True\n"
            "assert rl.allow_request('ip1', 5.0) is True\n"
            "assert rl.allow_request('ip1', 9.0) is False\n"
            "assert rl.allow_request('ip2', 9.0) is True\n"
            "assert rl.allow_request('ip1', 11.1) is True\n"
        ),
    ),
]


def evaluate_single_candidate(
    model_spec: str,
    task: BenchmarkTask,
    ollama_base_url: str = "http://127.0.0.1:11434",
    openai_base_url: str = "https://api.openai.com/v1",
    sandbox_timeout: float = 2.5,
) -> CandidateEvaluation:
    """Generate code from model_spec, validate via AST gate, and run sandboxed assertions."""
    raw_resp, infer_ms = generate_candidate_completion(
        model_spec=model_spec,
        task=task,
        ollama_base_url=ollama_base_url,
        openai_base_url=openai_base_url,
    )
    code = extract_python_code(raw_resp)
    ast_check = inspect_code_safety(code)

    if not ast_check.syntax_valid or not ast_check.is_safe:
        return CandidateEvaluation(
            model_id=model_spec,
            task_id=task.task_id,
            raw_response=raw_resp,
            extracted_code=code,
            syntax_valid=ast_check.syntax_valid,
            ast_safe=ast_check.is_safe,
            ast_violations=ast_check.violations,
            cyclomatic_complexity=ast_check.cyclomatic_complexity,
            inference_ms=infer_ms,
            error_output="; ".join(ast_check.violations),
        )

    exec_res = execute_unit_tests_sandboxed(code, task.test_harness, timeout_seconds=sandbox_timeout)
    return CandidateEvaluation(
        model_id=model_spec,
        task_id=task.task_id,
        raw_response=raw_resp,
        extracted_code=code,
        syntax_valid=True,
        ast_safe=True,
        ast_violations=[],
        cyclomatic_complexity=ast_check.cyclomatic_complexity,
        tests_passed=exec_res.passed,
        timed_out=exec_res.timed_out,
        execution_ms=exec_res.duration_ms,
        inference_ms=infer_ms,
        error_output=exec_res.error_output,
    )


def build_preference_pair(
    task: BenchmarkTask,
    eval_a: CandidateEvaluation,
    eval_b: CandidateEvaluation,
) -> PairwisePreferenceRecord:
    """Rank two candidate generations using deterministic RLHF code-quality criteria."""
    if eval_a.composite_score >= eval_b.composite_score:
        winner_eval, loser_eval = eval_a, eval_b
    else:
        winner_eval, loser_eval = eval_b, eval_a

    if eval_a.composite_score == eval_b.composite_score:
        if eval_a.cyclomatic_complexity <= eval_b.cyclomatic_complexity:
            winner_eval, loser_eval = eval_a, eval_b
        else:
            winner_eval, loser_eval = eval_b, eval_a

    reasons: list[str] = []
    if winner_eval.tests_passed and not loser_eval.tests_passed:
        reasons.append("passed hidden unit assertions while opponent failed")
    if winner_eval.ast_safe and not loser_eval.ast_safe:
        reasons.append("satisfied static AST security gate while opponent triggered policy violation")
    if not reasons:
        reasons.append(
            f"achieved higher rubric efficiency (complexity={winner_eval.cyclomatic_complexity}, exec={winner_eval.execution_ms}ms)"
        )

    return PairwisePreferenceRecord(
        task_id=task.task_id,
        prompt=task.prompt,
        model_a=eval_a.model_id,
        model_b=eval_b.model_id,
        winner=winner_eval.model_id,
        chosen=winner_eval.extracted_code,
        rejected=loser_eval.extracted_code,
        rationale="; ".join(reasons),
        score_a=eval_a.composite_score,
        score_b=eval_b.composite_score,
    )


def evaluate_pair_on_tasks(
    model_a: str,
    model_b: str,
    tasks: list[BenchmarkTask] | None = None,
    ollama_base_url: str = "http://127.0.0.1:11434",
    openai_base_url: str = "https://api.openai.com/v1",
) -> tuple[list[CandidateEvaluation], list[PairwisePreferenceRecord]]:
    """Run pairwise arena across all tasks and produce telemetry + DPO preference pairs."""
    suite = tasks if tasks is not None else DEFAULT_TASKS
    all_evals: list[CandidateEvaluation] = []
    preferences: list[PairwisePreferenceRecord] = []

    for task in suite:
        ev_a = evaluate_single_candidate(
            model_a, task, ollama_base_url=ollama_base_url, openai_base_url=openai_base_url
        )
        ev_b = evaluate_single_candidate(
            model_b, task, ollama_base_url=ollama_base_url, openai_base_url=openai_base_url
        )
        all_evals.extend([ev_a, ev_b])
        preferences.append(build_preference_pair(task, ev_a, ev_b))

    return all_evals, preferences
