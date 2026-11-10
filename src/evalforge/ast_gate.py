"""Pre-execution Python AST security gatekeeper and McCabe complexity analyzer."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re

BANNED_MODULES = frozenset(
    {
        "ctypes",
        "http",
        " importlib",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "pty",
        "resource",
        "shutil",
        "signal",
        "socket",
        "ssl",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "webbrowser",
    }
)

BANNED_BUILTINS = frozenset(
    {
        "__import__",
        " breakpoint",
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
    }
)

DUNDER_ESCAPE_ATTRS = frozenset(
    {"__bases__", "__builtins__", "__class__", "__code__", "__globals__", "__mro__", "__subclasses__"}
)

_FENCE_PATTERN = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_code(raw_response: str) -> str:
    """Extract clean Python source from LLM markdown code blocks or raw completion text."""
    matches = _FENCE_PATTERN.findall(raw_response)
    if matches:
        return matches[0].strip()
    return raw_response.strip()


@dataclass(frozen=True)
class ASTInspectionResult:
    """Result of static AST safety validation and cyclomatic complexity calculation."""

    syntax_valid: bool
    is_safe: bool
    violations: list[str]
    cyclomatic_complexity: int


class _SecurityAndComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.complexity: int = 1

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root_pkg = alias.name.split(".")[0]
            if root_pkg in BANNED_MODULES:
                self.violations.append(f"Disallowed module import: '{root_pkg}' (line {node.lineno})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.module:
            root_pkg = node.module.split(".")[0]
            if root_pkg in BANNED_MODULES:
                self.violations.append(f"Disallowed from-import: '{root_pkg}' (line {node.lineno})")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
            self.violations.append(f"Disallowed builtin call: '{node.func.id}()' (line {node.lineno})")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in DUNDER_ESCAPE_ATTRS:
            self.violations.append(
                f"Disallowed sandbox-escape dunder attribute: '{node.attr}' (line {node.lineno})"
            )
        self.generic_visit(node)

    def _increment_complexity(self, node: ast.AST) -> None:
        self.complexity += 1
        self.generic_visit(node)

    visit_If = _increment_complexity  # noqa: N815
    visit_For = _increment_complexity  # noqa: N815
    visit_AsyncFor = _increment_complexity  # noqa: N815
    visit_While = _increment_complexity  # noqa: N815
    visit_ExceptHandler = _increment_complexity  # noqa: N815
    visit_IfExp = _increment_complexity  # noqa: N815

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        self.complexity += max(0, len(node.values) - 1)
        self.generic_visit(node)


def inspect_code_safety(code: str) -> ASTInspectionResult:
    """Parse candidate Python code into an AST, checking safety rules and complexity."""
    if not code.strip():
        return ASTInspectionResult(
            syntax_valid=False,
            is_safe=False,
            violations=["Empty code completion."],
            cyclomatic_complexity=0,
        )
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ASTInspectionResult(
            syntax_valid=False,
            is_safe=False,
            violations=[f"SyntaxError: {exc.msg} (line {exc.lineno})"],
            cyclomatic_complexity=0,
        )

    visitor = _SecurityAndComplexityVisitor()
    visitor.visit(tree)
    return ASTInspectionResult(
        syntax_valid=True,
        is_safe=len(visitor.violations) == 0,
        violations=visitor.violations,
        cyclomatic_complexity=visitor.complexity,
    )
