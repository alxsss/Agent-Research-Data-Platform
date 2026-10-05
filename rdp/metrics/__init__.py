"""Run logging: metrics, transcripts, and grades, buffered and flushed to the store.

    with rdp.start_run(kind="eval", model="claude-sonnet") as run:
        run.log({"loss": 1.2}, step=10)
        ep = run.log_episode(turns=[...], tool_calls=[...], architecture="A")
        run.log_grade(ep, grader="exact", score=1.0)
"""

from __future__ import annotations

import atexit
import json
import subprocess
import uuid
from datetime import UTC, datetime
from typing import Self

from rdp.schemas import SCHEMAS
from rdp.store import SchemaError, Store

INLINE_LIMIT = 4096  # characters; longer turn content goes to the blob store


def _now() -> datetime:
    return datetime.now(UTC)


def _code_commit() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("-dirty" if dirty else "")


class Run:
    def __init__(self, store: Store, kind: str, model: str | None = None,
                 config: dict | None = None, run_id: str | None = None,
                 flush_every: int = 1000):
        self.store = store
        self.run_id = run_id or f"{kind}-{_now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.flush_every = flush_every
        self._buf: dict[str, list[dict]] = {}
        self._n_buffered = 0
        self._finished = False
        self._run_row = {
            "run_id": self.run_id, "kind": kind, "model": model,
            "config_json": json.dumps(config or {}, sort_keys=True, default=str),
            "code_commit": _code_commit(), "started_at": _now(),
        }
        self._write_status("running")
        atexit.register(self._atexit)

    # -------------------------------------------------------------- logging
    def log(self, metrics: dict[str, float], step: int) -> None:
        ts = _now()
        for name, value in metrics.items():
            self._add("metrics", {"run_id": self.run_id, "step": step, "name": name,
                                  "value": float(value), "ts": ts})

    def log_episode(self, turns: list[dict], tool_calls: list[dict] | None = None,
                    episode_id: str | None = None, **fields) -> str:
        """Log one transcript.

        turns: [{"role", "content", "token_count"?}] in order.
        tool_calls: [{"turn_idx", "tool_name", "args"?, "result"?, "result_tokens"?,
                      "duration_ms"?}]; results are stored as blobs.
        fields: other ``episodes`` columns (item_id, architecture, final_answer, ...).
        """
        episode_id = episode_id or uuid.uuid4().hex
        for idx, turn in enumerate(turns):
            content = turn.get("content")
            ref = None
            if content is not None and len(content) > INLINE_LIMIT:
                ref, content = self.store.put_blob(content), None
            self._add("turns", {"run_id": self.run_id, "episode_id": episode_id,
                                "turn_idx": idx, "role": turn["role"], "content": content,
                                "content_ref": ref, "token_count": turn.get("token_count")})
        call_counts: dict[int, int] = {}
        for call in tool_calls or []:
            t = call["turn_idx"]
            call_counts[t] = call_counts.get(t, 0) + 1
            result = call.get("result")
            self._add("tool_calls", {
                "run_id": self.run_id, "episode_id": episode_id, "turn_idx": t,
                "call_idx": call_counts[t] - 1, "tool_name": call["tool_name"],
                "args_json": json.dumps(call.get("args", {}), sort_keys=True),
                "result_ref": self.store.put_blob(result) if result is not None else None,
                "result_tokens": call.get("result_tokens"),
                "duration_ms": call.get("duration_ms"),
            })
        fields.setdefault("started_at", _now())
        self._add("episodes", {"run_id": self.run_id, "episode_id": episode_id,
                               "n_turns": len(turns), **fields})
        return episode_id

    def log_grade(self, episode_id: str, grader: str, score: float | None = None,
                  label: str | None = None, failure_mode: str | None = None) -> None:
        self._add("grades", {"run_id": self.run_id, "episode_id": episode_id,
                             "grader": grader, "score": score, "label": label,
                             "failure_mode": failure_mode, "graded_at": _now()})

    # ------------------------------------------------------------ lifecycle
    def flush(self) -> None:
        for table, rows in self._buf.items():
            self.store.write(table, rows, self.run_id)
        self._buf, self._n_buffered = {}, 0

    def finish(self, status: str = "ok") -> None:
        if self._finished:
            return
        self.flush()
        self._write_status(status)
        self._finished = True
        atexit.unregister(self._atexit)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish("failed" if exc_type else "ok")

    # ------------------------------------------------------------- internal
    def _add(self, table: str, row: dict) -> None:
        if self._finished:
            raise RuntimeError(f"run {self.run_id} is finished")
        extra = set(row) - set(SCHEMAS[table].names)
        if extra:  # fail at the call site, not at a later flush
            raise SchemaError(f"{table}: unknown columns {sorted(extra)}")
        self._buf.setdefault(table, []).append(row)
        self._n_buffered += 1
        if self._n_buffered >= self.flush_every:
            self.flush()

    def _write_status(self, status: str) -> None:
        self.store.write("runs", [{**self._run_row, "status": status, "updated_at": _now()}],
                         self.run_id)

    def _atexit(self) -> None:
        # Process exiting without finish(): keep the data, mark the run crashed.
        self.finish("crashed")


def start_run(kind: str, model: str | None = None, config: dict | None = None,
              store: Store | None = None, **kwargs) -> Run:
    return Run(store or Store(), kind=kind, model=model, config=config, **kwargs)
