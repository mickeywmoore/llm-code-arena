"""Unit and ephemeral mock-Ollama HTTP integration tests for EvalForge."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from evalforge.ast_gate import inspect_code_safety
from evalforge.cli import main
from evalforge.sandbox import execute_unit_tests_sandboxed


class _MockOllamaHandler(BaseHTTPRequestHandler):
    """Simulates local Ollama /api/generate endpoint serving competing model completions."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        model = body.get("model", "")
        prompt = body.get("prompt", "")

        if "merge_intervals" in prompt:
            code = (
                "```python\n"
                "def merge_intervals(intervals):\n"
                "    if not intervals: return []\n"
                "    s = sorted(intervals)\n"
                "    out = [s[0]]\n"
                "    for a, b in s[1:]:\n"
                "        if a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))\n"
                "        else: out.append((a, b))\n"
                "    return out\n"
                "```"
                if model == "coder-strong"
                else "```python\nimport subprocess\ndef merge_intervals(x): subprocess.run(['id'])\n```"
            )
        elif "LRUCache" in prompt:
            code = (
                "```python\n"
                "from collections import OrderedDict\n"
                "class LRUCache:\n"
                "    def __init__(self, c):\n"
                "        self.c, self.d = c, OrderedDict()\n"
                "    def get(self, k):\n"
                "        if k not in self.d: return -1\n"
                "        self.d.move_to_end(k)\n"
                "        return self.d[k]\n"
                "    def put(self, k, v):\n"
                "        if k in self.d: self.d.move_to_end(k)\n"
                "        self.d[k] = v\n"
                "        if len(self.d) > self.c: self.d.popitem(last=False)\n"
                "```"
                if model == "coder-strong"
                else "```python\nclass LRUCache:\n    def __init__(self, c): pass\n    def get(self, k): return 0\n    def put(self, k, v): pass\n```"
            )
        else:
            code = (
                "```python\n"
                "from collections import deque\n"
                "class SlidingWindowRateLimiter:\n"
                "    def __init__(self, m, w):\n"
                "        self.m, self.w, self.h = m, w, {}\n"
                "    def allow_request(self, c, t):\n"
                "        q = self.h.setdefault(c, deque())\n"
                "        while q and q[0] <= t - self.w: q.popleft()\n"
                "        if len(q) >= self.m: return False\n"
                "        q.append(t)\n"
                "        return True\n"
                "```"
                if model == "coder-strong"
                else "```python\nclass SlidingWindowRateLimiter:\n    def __init__(self, m, w): pass\n    def allow_request(self, c, t): return False\n```"
            )

        resp_bytes = json.dumps({"response": code, "done": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


class TestEvalForgeArena(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
        cls.port = cls.server.server_port
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def test_ast_gate_blocks_unsafe_imports_eval_and_dunders(self) -> None:
        unsafe = "import os\ndef f(x):\n    return eval(x).__subclasses__()\n"
        res = inspect_code_safety(unsafe)
        self.assertTrue(res.syntax_valid)
        self.assertFalse(res.is_safe)
        self.assertGreaterEqual(len(res.violations), 3)

    def test_sandbox_enforces_wall_clock_timeout(self) -> None:
        infinite_loop = "def hang():\n    while True:\n        pass\n"
        res = execute_unit_tests_sandboxed(infinite_loop, "hang()", timeout_seconds=0.3)
        self.assertFalse(res.passed)
        self.assertTrue(res.timed_out)

    def test_end_to_end_ollama_wire_protocol_and_dpo_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dpo_file = Path(tmpdir) / "prefs.jsonl"
            json_file = Path(tmpdir) / "summary.json"
            exit_code = main(
                [
                    "arena",
                    "--model-a",
                    "ollama:coder-strong",
                    "--model-b",
                    "ollama:coder-weak",
                    "--ollama-url",
                    f"http://127.0.0.1:{self.port}",
                    "--export-dpo-jsonl",
                    str(dpo_file),
                    "--output-json",
                    str(json_file),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(dpo_file.exists())
            lines = dpo_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            first_record = json.loads(lines[0])
            self.assertEqual(first_record["winner"], "ollama:coder-strong")
            self.assertIn("AST security gate", first_record["rationale"])


if __name__ == "__main__":
    unittest.main()
