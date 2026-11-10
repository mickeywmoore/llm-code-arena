# EvalForge: LLM Code Generation Arena, AST Safety Gate & RLHF Evaluator

[![EvalForge CI Pipeline](https://github.com/mickeywmoore/llm-code-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/mickeywmoore/llm-code-arena/actions/workflows/ci.yml)

A **zero-dependency** Python CLI (`evalforge`) for deterministic side-by-side evaluation of local (**Ollama**) and cloud (**OpenAI-compatible**) code-generation models using static **Python AST security gating**, isolated subprocess test sandboxing, McCabe complexity scoring, and **RLHF / DPO preference JSONL** serialization.

---

## Systems Architecture & Engineering Trade-Offs

1. **Pre-Execution Python AST Security Gate (`src/evalforge/ast_gate.py`)**
   * Parses untrusted LLM completions into Abstract Syntax Trees (`ast.parse`) *before* execution, blocking dangerous module imports (`os`, `subprocess`, `socket`, `urllib`, `ctypes`), dynamic code-execution builtins (`eval`, `exec`, `__import__`), and dunder sandbox-escape chains (`__subclasses__`, `__globals__`) while computing McCabe cyclomatic complexity.
2. **Isolated Subprocess Execution Sandbox (`src/evalforge/sandbox.py`)**
   * Executes AST-vetted candidate code against hidden unit-test assertions inside an isolated Python interpreter (`python -I -B`) in an ephemeral workspace with stripped `PATH` and strict wall-clock timeouts.
3. **Zero-SDK Local & Cloud LLM Wire Protocols (`src/evalforge/providers.py`)**
   * Communicates directly with local **Ollama** daemons (`/api/generate`) and **OpenAI-compatible** REST endpoints (`/v1/chat/completions`) via standard-library `urllib.request` with deterministic `temperature=0.0` seeding.
4. **Automated RLHF / DPO Preference Ranking (`src/evalforge/arena.py`)**
   * Ranks competing completions across correctness (`Pass@1`), AST policy compliance, cyclomatic complexity, and runtime latency, exporting ready-to-train `{prompt, chosen, rejected, rationale}` JSONL datasets.

---

## Repository Structure

```text
llm-code-arena/
├── .github/workflows/ci.yml  # Python 3.11 & 3.12 CI test matrix
├── pyproject.toml            # Zero-dependency PEP 621 package config
├── README.md
├── src/evalforge/
│   ├── __init__.py
│   ├── arena.py              # Benchmark suite & pairwise RLHF/DPO ranker
│   ├── ast_gate.py           # Static AST security & McCabe complexity visitor
│   ├── cli.py                # CLI entry point (evalforge arena)
│   ├── models.py             # Typed evaluation & preference dataclasses
│   ├── providers.py          # Ollama & OpenAI REST clients + offline CI mocks
│   └── sandbox.py            # Isolated subprocess runner with timeout enforcement
└── tests/
    └── test_evalforge.py     # AST security, timeout & ephemeral Ollama HTTP tests
```

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Benchmark two local Ollama models head-to-head and export RLHF/DPO JSONL pairs
evalforge arena --model-a ollama:qwen2.5-coder --model-b ollama:llama3 --export-dpo-jsonl artifacts/dpo_pairs.jsonl

# Run deterministic offline arena verification
evalforge arena --model-a mock:optimal --model-b mock:flawed --output-json artifacts/arena.json

# Execute unit & ephemeral Ollama wire-protocol test suite
python -m unittest discover -s tests -v
```
