"""The agent loop: a model answers one question by calling one setup's search tools.

Works with any OpenAI-compatible chat API: OpenRouter (Claude and others) or a
local vLLM server (the small model in Week 3). The loop is:

    question -> model -> tool calls? -> run tools -> results back -> model -> ...
                      -> no tool calls -> final answer

``run_agent`` returns everything ``rdp.Run.log_episode`` needs, so every message,
tool call, and result ends up in the platform.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from research.tools import Architecture

OPENROUTER_URL = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = """\
You answer questions about films, TV and music using only the search tools provided, \
which cover a collection of Wikipedia articles. Always search before answering, and \
base your answer only on what the tools return, not on your own memory. If the \
collection does not contain the answer, say so.

When you are done, end your reply with one line in this exact form:
Answer: <short answer, a few words, or "I don't know">"""

# Accepts "Answer: x", "**Answer:** x", "**Answer**: x", "Answer: **x**".
ANSWER_LINE = re.compile(r"^\s*\**answer\**\s*:\s*\**\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass
class AgentResult:
    final_answer: str | None
    stop_reason: str                  # answered | max_steps | error:<type>
    turns: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    n_model_calls: int = 0


def make_client(base_url: str = OPENROUTER_URL, api_key_env: str = "OPENROUTER_API_KEY"):
    """OpenAI-compatible client. The key is read from the environment, never passed in."""
    from openai import OpenAI

    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"Set {api_key_env} in your shell first.")
    return OpenAI(base_url=base_url, api_key=key, max_retries=5, timeout=120)


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    matches = ANSWER_LINE.findall(text)
    return (matches[-1] if matches else text).strip().strip("*").strip()


def run_agent(client, model: str, arch: Architecture, question: str, max_steps: int = 10,
              temperature: float = 0.0, system_prompt: str = SYSTEM_PROMPT) -> AgentResult:
    """Run one episode. Never raises on API errors: they become stop_reason="error:..."."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    res = AgentResult(final_answer=None, stop_reason="max_steps",
                      turns=[{"role": "user", "content": question}])
    start = time.perf_counter()
    try:
        for _ in range(max_steps):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=list(arch.tools),
                temperature=temperature)
            res.n_model_calls += 1
            if resp.usage is not None:
                res.tokens_in += resp.usage.prompt_tokens or 0
                res.tokens_out += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            calls = msg.tool_calls or []

            res.turns.append({"role": "assistant", "content": msg.content,
                              "token_count": resp.usage.completion_tokens if resp.usage
                              else None})
            assistant_turn = len(res.turns) - 1
            if not calls:
                res.final_answer = extract_answer(msg.content)
                res.stop_reason = "answered"
                break

            messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls]})
            for c in calls:
                t0 = time.perf_counter()
                result = arch.call(c.function.name, c.function.arguments)
                res.tool_calls.append({
                    "turn_idx": assistant_turn, "tool_name": c.function.name,
                    "args": _safe_args(c.function.arguments), "result": result,
                    "duration_ms": (time.perf_counter() - t0) * 1000})
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
                res.turns.append({"role": "tool", "content": result})
    except Exception as e:  # noqa: BLE001 -- any API failure ends this episode, not the run
        res.stop_reason = f"error:{type(e).__name__}"
        res.turns.append({"role": "error", "content": str(e)[:2000]})
    res.latency_ms = (time.perf_counter() - start) * 1000
    return res


def _safe_args(arguments: str | None) -> dict:
    try:
        parsed = json.loads(arguments or "{}")
        return parsed if isinstance(parsed, dict) else {"_raw": arguments}
    except json.JSONDecodeError:
        return {"_raw": arguments}
