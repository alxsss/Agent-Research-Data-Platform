"""Canonical Arrow schemas for every table the platform stores.

Versioning rule: within a major version, changes are additive only (new nullable
columns). Readers use DuckDB ``union_by_name`` so files written under older minor
versions remain queryable. A major bump requires a migration script.
"""

import pyarrow as pa

SCHEMA_VERSION = "1.0"

TS = pa.timestamp("us", tz="UTC")


def _f(name, typ, nullable=True):
    return pa.field(name, typ, nullable=nullable)


SCHEMAS: dict[str, pa.Schema] = {
    # Append-only: a status change is a new row; the latest updated_at wins.
    "runs": pa.schema([
        _f("run_id", pa.string(), False),
        _f("kind", pa.string(), False),          # eval | sft | grpo | datagen | ...
        _f("model", pa.string()),
        _f("config_json", pa.string()),
        _f("code_commit", pa.string()),
        _f("status", pa.string(), False),        # running | ok | failed | crashed
        _f("started_at", TS, False),
        _f("updated_at", TS, False),
    ]),
    "metrics": pa.schema([
        _f("run_id", pa.string(), False),
        _f("step", pa.int64(), False),
        _f("name", pa.string(), False),
        _f("value", pa.float64(), False),
        _f("ts", TS, False),
    ]),
    "episodes": pa.schema([
        _f("run_id", pa.string(), False),
        _f("episode_id", pa.string(), False),
        _f("dataset_version", pa.string()),
        _f("item_id", pa.string()),
        _f("architecture", pa.string()),
        _f("final_answer", pa.string()),
        _f("n_turns", pa.int32(), False),
        _f("tokens_in", pa.int64()),
        _f("tokens_out", pa.int64()),
        _f("latency_ms", pa.float64()),
        _f("started_at", TS, False),
    ]),
    # Long content is stored in the blob store; exactly one of content/content_ref is set.
    "turns": pa.schema([
        _f("run_id", pa.string(), False),
        _f("episode_id", pa.string(), False),
        _f("turn_idx", pa.int32(), False),
        _f("role", pa.string(), False),
        _f("content", pa.string()),
        _f("content_ref", pa.string()),
        _f("token_count", pa.int32()),
    ]),
    "tool_calls": pa.schema([
        _f("run_id", pa.string(), False),
        _f("episode_id", pa.string(), False),
        _f("turn_idx", pa.int32(), False),
        _f("call_idx", pa.int32(), False),
        _f("tool_name", pa.string(), False),
        _f("args_json", pa.string()),
        _f("result_ref", pa.string()),
        _f("result_tokens", pa.int32()),
        _f("duration_ms", pa.float64()),
    ]),
    # Corrections (e.g. a human re-grade) are new rows; the latest graded_at per
    # (episode_id, grader) wins.
    "grades": pa.schema([
        _f("run_id", pa.string(), False),
        _f("episode_id", pa.string(), False),
        _f("grader", pa.string(), False),        # exact | f1 | llm_judge | human
        _f("score", pa.float64()),
        _f("label", pa.string()),
        _f("failure_mode", pa.string()),
        _f("graded_at", TS, False),
    ]),
}

TABLES = tuple(SCHEMAS)


def major(version: str) -> int:
    return int(version.split(".")[0])
