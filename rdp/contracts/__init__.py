"""Data contracts: checks that must pass before data is trusted.

Two kinds:
- run checks (``check_run``) validate a run's transcript tables in the store;
- dataset checks (e.g. ``no_overlap``) are passed to ``Registry.register(checks=...)``
  and block registration on failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from rdp.schemas import SCHEMA_VERSION, major

EXAMPLE_LIMIT = 5


@dataclass
class CheckResult:
    name: str
    passed: bool
    n_violations: int = 0
    examples: list = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if self.passed else f"FAIL ({self.n_violations})"
        ex = f"  e.g. {self.examples}" if self.examples else ""
        return f"{status:<12} {self.name}{ex}"


class ContractViolation(Exception):
    def __init__(self, results: list[CheckResult]):
        self.results = results
        super().__init__("; ".join(str(r) for r in results))


# --------------------------------------------------------------- run checks
# Each query returns one row per violation for the run bound to $run_id.
RUN_CHECKS: dict[str, str] = {
    "tool_calls_reference_existing_turn": """
        SELECT tc.episode_id, tc.turn_idx FROM tool_calls tc
        ANTI JOIN turns t USING (episode_id, turn_idx)
        WHERE tc.run_id = $run_id""",
    "every_episode_has_a_grade": """
        SELECT e.episode_id FROM episodes e
        ANTI JOIN grades_history g USING (episode_id)
        WHERE e.run_id = $run_id""",
    "token_counts_non_negative": """
        SELECT episode_id FROM episodes
        WHERE run_id = $run_id AND (tokens_in < 0 OR tokens_out < 0)
        UNION ALL
        SELECT episode_id FROM turns WHERE run_id = $run_id AND token_count < 0""",
    "turn_idx_contiguous_from_zero": """
        SELECT episode_id FROM turns WHERE run_id = $run_id
        GROUP BY episode_id
        HAVING min(turn_idx) <> 0 OR max(turn_idx) + 1 <> count(*)
            OR count(DISTINCT turn_idx) <> count(*)""",
    "n_turns_matches_turns": """
        SELECT e.episode_id FROM episodes e
        LEFT JOIN (SELECT episode_id, count(*) AS n FROM turns WHERE run_id = $run_id
                   GROUP BY episode_id) t USING (episode_id)
        WHERE e.run_id = $run_id AND e.n_turns <> coalesce(t.n, 0)""",
    "turn_has_exactly_one_content_source": """
        SELECT episode_id, turn_idx FROM turns
        WHERE run_id = $run_id AND content IS NOT NULL AND content_ref IS NOT NULL""",
}


def _schema_version_check(store, run_id: str) -> CheckResult:
    bad = []
    for table in ("runs", "metrics", "episodes", "turns", "tool_calls", "grades"):
        for f in store.files(table, run_id):
            meta = pq.read_schema(f).metadata or {}
            v = meta.get(b"rdp.schema_version", b"").decode()
            if not v or major(v) != major(SCHEMA_VERSION):
                bad.append(f"{table}/{f.name}: {v or 'missing'}")
    return CheckResult("schema_major_version_matches", not bad, len(bad), bad[:EXAMPLE_LIMIT])


def check_run(store, run_id: str, con: duckdb.DuckDBPyConnection | None = None
              ) -> list[CheckResult]:
    from rdp.query import connect

    con = con or connect(store)
    results = []
    for name, sql in RUN_CHECKS.items():
        rows = con.execute(sql, {"run_id": run_id}).fetchall()
        results.append(CheckResult(name, not rows, len(rows), rows[:EXAMPLE_LIMIT]))
    results.append(_schema_version_check(store, run_id))
    return results


# ----------------------------------------------------------- dataset checks
def _reader(path: Path) -> str:
    return {".parquet": "read_parquet", ".csv": "read_csv_auto"}.get(
        path.suffix.lower(), "read_json_auto")


def no_overlap(other: str | Path, key: str, other_key: str | None = None,
               name: str = "no_train_eval_overlap") -> Callable[[Path], CheckResult]:
    """Dataset check: no value of ``key`` in the new dataset appears in ``other``.

    Values are compared after lower-casing and whitespace normalization, which
    catches the common contamination case of the same question re-generated.
    """
    other = Path(other)
    other_key = other_key or key

    def norm(col: str) -> str:
        return f"regexp_replace(lower(trim(CAST({col} AS VARCHAR))), '\\s+', ' ', 'g')"

    def check(path: Path) -> CheckResult:
        with duckdb.connect() as con:
            rows = con.execute(f"""
                SELECT DISTINCT {norm(f'a."{key}"')} FROM {_reader(path)}(?) a
                SEMI JOIN {_reader(other)}(?) b
                ON {norm(f'a."{key}"')} = {norm(f'b."{other_key}"')}
            """, [str(path), str(other)]).fetchall()
        return CheckResult(name, not rows, len(rows), [r[0] for r in rows[:EXAMPLE_LIMIT]])

    return check
