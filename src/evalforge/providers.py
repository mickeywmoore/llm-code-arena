"""Zero-dependency LLM inference adapters for Ollama, OpenAI-compatible APIs, and CI mocks."""

from __future__ import annotations

import json
import os
import time
import urllib.request

from evalforge.models import BenchmarkTask


MOCK_OPTIMAL_SOLUTIONS: dict[str, str] = {
    "ALG-001": (
        "```python\n"
        "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals, key=lambda pair: pair[0])\n"
        "    merged = [ordered[0]]\n"
        "    for start, end in ordered[1:]:\n"
        "        prev_start, prev_end = merged[-1]\n"
        "        if start <= prev_end:\n"
        "            merged[-1] = (prev_start, max(prev_end, end))\n"
        "        else:\n"
        "            merged.append((start, end))\n"
        "    return merged\n"
        "```"
    ),
    "ALG-002": (
        "```python\n"
        "from collections import OrderedDict\n\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int) -> None:\n"
        "        self.capacity = capacity\n"
        "        self._store: OrderedDict[int, int] = OrderedDict()\n\n"
        "    def get(self, key: int) -> int:\n"
        "        if key not in self._store:\n"
        "            return -1\n"
        "        self._store.move_to_end(key)\n"
        "        return self._store[key]\n\n"
        "    def put(self, key: int, value: int) -> None:\n"
        "        if key in self._store:\n"
        "            self._store.move_to_end(key)\n"
        "        self._store[key] = value\n"
        "        if len(self._store) > self.capacity:\n"
        "            self._store.popitem(last=False)\n"
        "```"
    ),
    "ALG-003": (
        "```python\n"
        "from collections import deque\n\n"
        "class SlidingWindowRateLimiter:\n"
        "    def __init__(self, max_requests: int, window_seconds: float) -> None:\n"
        "        self.max_requests = max_requests\n"
        "        self.window_seconds = window_seconds\n"
        "        self._hits: dict[str, deque[float]] = {}\n\n"
        "    def allow_request(self, client_id: str, timestamp: float) -> bool:\n"
        "        bucket = self._hits.setdefault(client_id, deque())\n"
        "        cutoff = timestamp - self.window_seconds\n"
        "        while bucket and bucket[0] <= cutoff:\n"
        "            bucket.popleft()\n"
        "        if len(bucket) >= self.max_requests:\n"
        "            return False\n"
        "        bucket.append(timestamp)\n"
        "        return True\n"
        "```"
    ),
}

MOCK_FLAWED_SOLUTIONS: dict[str, str] = {
    "ALG-001": (
        "```python\n"
        "def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
        "    # Buggy: fails to sort unsorted input intervals before merging\n"
        "    if not intervals:\n"
        "        return []\n"
        "    merged = [intervals[0]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= merged[-1][1]:\n"
        "            merged[-1] = (merged[-1][0], max(merged[-1][1], e))\n"
        "        else:\n"
        "            merged.append((s, e))\n"
        "    return merged\n"
        "```"
    ),
    "ALG-002": (
        "```python\n"
        "import os\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int) -> None:\n"
        "        os.system('echo unsafe')\n"
        "```"
    ),
    "ALG-003": (
        "```python\n"
        "class SlidingWindowRateLimiter:\n"
        "    def __init__(self, max_requests: int, window_seconds: float) -> None:\n"
        "        self.max_requests = max_requests\n"
        "    def allow_request(self, client_id: str, timestamp: float) -> bool:\n"
        "        return True\n"
        "```"
    ),
}


def generate_candidate_completion(
    model_spec: str,
    task: BenchmarkTask,
    ollama_base_url: str = "http://127.0.0.1:11434",
    openai_base_url: str = "https://api.openai.com/v1",
    timeout: float = 45.0,
) -> tuple[str, float]:
    """Query model_spec ('ollama:<tag>', 'openai:<model>', or 'mock:<tier>') and return (text, ms)."""
    system_instruction = (
        "Return ONLY valid Python 3 standard-library code implementing the requested interface. "
        "Do not import os, sys, subprocess, socket, or external third-party packages."
    )
    full_prompt = f"{system_instruction}\n\nTask:\n{task.prompt}"
    t0 = time.perf_counter()

    if model_spec.startswith("mock:"):
        tier = model_spec.split(":", 1)[1]
        bank = MOCK_OPTIMAL_SOLUTIONS if tier == "optimal" else MOCK_FLAWED_SOLUTIONS
        completion = bank.get(task.task_id, "def placeholder(): pass")
        return completion, round((time.perf_counter() - t0) * 1000.0, 2)

    if model_spec.startswith("ollama:"):
        model_name = model_spec.split(":", 1)[1]
        payload = json.dumps(
            {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "seed": 42},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_base_url.rstrip('/')}/api/generate",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return str(body.get("response", "")), round((time.perf_counter() - t0) * 1000.0, 2)

    if model_spec.startswith("openai:"):
        model_name = model_spec.split(":", 1)[1]
        api_key = os.environ.get("OPENAI_API_KEY", "")
        payload = json.dumps(
            {
                "model": model_name,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": task.prompt},
                ],
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(
            f"{openai_base_url.rstrip('/')}/chat/completions",
            data=payload,
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return str(content), round((time.perf_counter() - t0) * 1000.0, 2)

    raise ValueError(
        f"Unsupported model spec '{model_spec}'. Expected 'ollama:<model>', 'openai:<model>', or 'mock:<tier>'."
    )
