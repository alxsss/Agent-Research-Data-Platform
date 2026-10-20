"""Dataset registry: versions, content hashes, and lineage, stored in a DuckDB catalog.

    reg = Registry(store)
    v = reg.register("evalset", "data/evalset.jsonl")
    sft = reg.register("sft", "data/sft.parquet", parents=[v], producer_run_id=run.run_id,
                       checks=[lambda p: no_overlap(...)])
    reg.lineage(sft)   # ancestors, nearest first
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from rdp.contracts import CheckResult, ContractViolation
from rdp.metrics import _code_commit
from rdp.store import Store

_DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    version_id      VARCHAR PRIMARY KEY,   -- "<name>:v<n>"
    name            VARCHAR NOT NULL,
    version         INTEGER NOT NULL,
    content_hash    VARCHAR NOT NULL,
    path            VARCHAR NOT NULL,
    format          VARCHAR,
    row_count       BIGINT,
    schema_json     VARCHAR,
    code_commit     VARCHAR,
    producer_run_id VARCHAR,
    description     VARCHAR,
    created_at      TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS lineage (
    child_id  VARCHAR NOT NULL,
    parent_id VARCHAR NOT NULL
);
"""


@dataclass(frozen=True)
class DatasetVersion:
    version_id: str
    name: str
    version: int
    content_hash: str
    path: str
    row_count: int | None


def content_hash(path: Path) -> str:
    """sha256 over bytes (and relative names inside a directory, so renames there count)."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        if path.is_dir():
            h.update(str(f.relative_to(path)).encode() + b"\0")
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _describe(path: Path) -> tuple[str | None, int | None, str | None]:
    """Return (format, row_count, schema_json) for formats DuckDB can read.

    A directory is described only if all its files share one readable format;
    anything else (e.g. a model checkpoint or a search index) gets (None, None, None).
    """
    if path.is_dir():
        suffixes = {p.suffix.lower() for p in path.rglob("*") if p.is_file()}
        if len(suffixes) != 1:
            return None, None, None
        suffix = suffixes.pop()
        target = str(path / "**" / f"*{suffix}")
    else:
        suffix, target = path.suffix.lower(), str(path)
    reader = {".parquet": "read_parquet", ".jsonl": "read_json_auto",
              ".json": "read_json_auto", ".csv": "read_csv_auto"}.get(suffix)
    if reader is None:
        return None, None, None
    with duckdb.connect() as con:
        cols = con.execute(f"DESCRIBE SELECT * FROM {reader}(?)", [target]).fetchall()
        n = con.execute(f"SELECT count(*) FROM {reader}(?)", [target]).fetchone()[0]
    return suffix.lstrip("."), n, json.dumps([[c[0], c[1]] for c in cols])


class Registry:
    def __init__(self, store: Store):
        self.path = store.root / "catalog.duckdb"
        with self._con() as con:
            con.execute(_DDL)

    def _con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def register(self, name: str, path: str | Path, parents: Iterable[str] = (),
                 producer_run_id: str | None = None, description: str | None = None,
                 checks: Iterable[Callable[[Path], CheckResult]] = ()) -> str:
        """Register a dataset version and return its version_id.

        Idempotent: registering identical content under the same name returns the
        existing version. Every check must pass or ContractViolation is raised and
        nothing is recorded.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        parents = list(parents)

        failed = [r for r in (check(path) for check in checks) if not r.passed]
        if failed:
            raise ContractViolation(failed)

        digest = content_hash(path)
        with self._con() as con:
            missing = [p for p in parents if not con.execute(
                "SELECT 1 FROM datasets WHERE version_id = ?", [p]).fetchone()]
            if missing:
                raise KeyError(f"unknown parent versions: {missing}")
            existing = con.execute(
                "SELECT version_id FROM datasets WHERE name = ? AND content_hash = ?",
                [name, digest]).fetchone()
            if existing:
                return existing[0]
            version = con.execute("SELECT coalesce(max(version), 0) + 1 FROM datasets "
                                  "WHERE name = ?", [name]).fetchone()[0]
            version_id = f"{name}:v{version}"
            fmt, rows, schema = _describe(path)
            con.execute("BEGIN")
            con.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [version_id, name, version, digest, str(path), fmt, rows, schema,
                 _code_commit(), producer_run_id, description, datetime.now(UTC)])
            for p in parents:
                con.execute("INSERT INTO lineage VALUES (?, ?)", [version_id, p])
            con.execute("COMMIT")
        return version_id

    def get(self, version_id: str) -> DatasetVersion:
        with self._con() as con:
            row = con.execute("SELECT version_id, name, version, content_hash, path, row_count "
                              "FROM datasets WHERE version_id = ?", [version_id]).fetchone()
        if row is None:
            raise KeyError(version_id)
        return DatasetVersion(*row)

    def verify(self, version_id: str) -> bool:
        """True if the files on disk still match the registered content hash."""
        v = self.get(version_id)
        p = Path(v.path)
        return p.exists() and content_hash(p) == v.content_hash

    def lineage(self, version_id: str) -> list[tuple[str, int, str | None]]:
        """Ancestors of a version as (version_id, depth, producer_run_id), nearest first."""
        self.get(version_id)
        with self._con() as con:
            return con.execute("""
                WITH RECURSIVE anc(id, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT l.parent_id, a.depth + 1
                    FROM lineage l JOIN anc a ON l.child_id = a.id
                )
                SELECT a.id, min(a.depth) AS depth, any_value(d.producer_run_id)
                FROM anc a JOIN datasets d ON d.version_id = a.id
                GROUP BY a.id ORDER BY depth, a.id
            """, [version_id]).fetchall()

    def list(self) -> list[tuple]:
        with self._con() as con:
            return con.execute("SELECT version_id, row_count, producer_run_id, created_at "
                               "FROM datasets ORDER BY name, version").fetchall()
