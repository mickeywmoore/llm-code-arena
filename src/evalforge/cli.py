"""Command-line interface for EvalForge pairwise LLM code arena & RLHF export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalforge.arena import DEFAULT_TASKS, evaluate_pair_on_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalforge",
        description="Zero-dependency LLM code-generation arena, AST sandbox gate, and RLHF preference exporter.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    arena_cmd = subparsers.add_parser(
        "arena",
        help="Run side-by-side coding benchmark between two LLM specs (ollama:<tag>, openai:<id>, or mock:<tier>).",
    )
    arena_cmd.add_argument(
        "--model-a",
        type=str,
        default="mock:optimal",
        help="Primary candidate model spec, e.g. 'ollama:qwen2.5-coder' or 'mock:optimal'.",
    )
    arena_cmd.add_argument(
        "--model-b",
        type=str,
        default="mock:flawed",
        help="Challenger model spec, e.g. 'ollama:llama3' or 'mock:flawed'.",
    )
    arena_cmd.add_argument(
        "--ollama-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Base URL for local Ollama daemon (default: http://127.0.0.1:11434).",
    )
    arena_cmd.add_argument(
        "--export-dpo-jsonl",
        type=str,
        default="",
        help="Optional path to export RLHF / DPO preference pairs as JSON Lines (.jsonl).",
    )
    arena_cmd.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to export full benchmark telemetry summary as JSON.",
    )
    return parser


def run_arena(args: argparse.Namespace) -> int:
    evals, prefs = evaluate_pair_on_tasks(
        model_a=args.model_a,
        model_b=args.model_b,
        tasks=DEFAULT_TASKS,
        ollama_base_url=args.ollama_url,
    )

    evals_by_model: dict[str, list] = {args.model_a: [], args.model_b: []}
    for ev in evals:
        evals_by_model.setdefault(ev.model_id, []).append(ev)

    print("=== EvalForge LLM Code Generation & AST Safety Arena ===")
    print(f"Benchmark Tasks : {len(DEFAULT_TASKS)} | Contenders: {args.model_a} vs. {args.model_b}")

    for m_id, m_evals in evals_by_model.items():
        pass_count = sum(1 for e in m_evals if e.tests_passed)
        safe_count = sum(1 for e in m_evals if e.ast_safe)
        avg_score = sum(e.composite_score for e in m_evals) / max(1, len(m_evals))
        print(
            f"\nModel: {m_id:22s} | Pass@1: {pass_count}/{len(m_evals)} ({pass_count / len(m_evals) * 100:.1f}%) "
            f"| AST Safe: {safe_count}/{len(m_evals)} | Avg Rubric Score: {avg_score:.2f}/100"
        )
        for e in m_evals:
            status = "PASS" if e.tests_passed else ("UNSAFE-AST" if not e.ast_safe else "FAIL")
            print(
                f"  [{status:10s}] {e.task_id} | Complexity: {e.cyclomatic_complexity:2d} "
                f"| Exec: {e.execution_ms:6.2f}ms | Score: {e.composite_score:6.2f}"
            )

    print("\n--- Pairwise RLHF / DPO Decisions ---")
    for pref in prefs:
        print(f"  {pref.task_id}: Winner -> {pref.winner} ({pref.rationale})")

    if args.export_dpo_jsonl:
        dpo_path = Path(args.export_dpo_jsonl)
        dpo_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(p.to_dpo_dict()) for p in prefs]
        dpo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nExported {len(prefs)} DPO preference records -> {dpo_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "tasks_evaluated": len(DEFAULT_TASKS),
            "evaluations": [e.to_dict() for e in evals],
            "pairwise_preferences": [p.to_dpo_dict() for p in prefs],
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved JSON telemetry report -> {json_path}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "arena":
        return run_arena(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
