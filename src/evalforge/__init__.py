"""EvalForge: Zero-dependency LLM Code Generation Arena, AST Gate & RLHF Evaluator."""

from evalforge.arena import DEFAULT_TASKS, evaluate_pair_on_tasks
from evalforge.ast_gate import ASTInspectionResult, extract_python_code, inspect_code_safety
from evalforge.models import BenchmarkTask, CandidateEvaluation, PairwisePreferenceRecord
from evalforge.sandbox import execute_unit_tests_sandboxed

__all__ = [
    "ASTInspectionResult",
    "BenchmarkTask",
    "CandidateEvaluation",
    "DEFAULT_TASKS",
    "PairwisePreferenceRecord",
    "evaluate_pair_on_tasks",
    "execute_unit_tests_sandboxed",
    "extract_python_code",
    "inspect_code_safety",
]
__version__ = "1.0.0"
