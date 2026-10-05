"""Append-only Parquet table store plus a content-addressed blob store.

Layout under the store root:

    tables/<table>/run_id=<run_id>/part-<timestamp>-<uuid>.parquet
    blobs/<sha256[:2]>/<sha256>
"""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from rdp.schemas import SCHEMA_VERSION, SCHEMAS

META_TABLE = b"rdp.table"
META_VERSION = b"rdp.schema_version"


class SchemaError(ValueError):
    pass


def default_root() -> Path:
    return Path(os.environ.get("RDP_HOME", "rdp_data"))


class Store:
    def __init__(self, root: str | os.PathLike | None = None):
        self.root = Path(root) if root is not None else default_root()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- tables
    def table_dir(self, table: str) -> Path:
        return self.root / "tables" / table

    def table_glob(self, table: str) -> str:
        return str(self.table_dir(table) / "**" / "*.parquet")

    def files(self, table: str, run_id: str | None = None) -> list[Path]:
        base = self.table_dir(table)
        if run_id is not None:
            base = base / f"run_id={run_id}"
        return sorted(base.rglob("*.parquet")) if base.exists() else []

    def write(self, table: str, rows: list[dict], run_id: str) -> Path | None:
        """Validate rows against the table schema and write them as one new file.

        Raises SchemaError on unknown columns, missing required values, or a
        run_id that does not match the partition.
        """
        if not rows:
            return None
        schema = SCHEMAS[table]
        known = set(schema.names)
        for row in rows:
            extra = set(row) - known
            if extra:
                raise SchemaError(f"{table}: unknown columns {sorted(extra)}")
            if row.get("run_id") != run_id:
                raise SchemaError(f"{table}: row run_id {row.get('run_id')!r} != {run_id!r}")
        try:
            t = pa.Table.from_pylist(rows, schema=schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            raise SchemaError(f"{table}: {e}") from e
        for field in schema:
            if not field.nullable and t.column(field.name).null_count:
                raise SchemaError(f"{table}: required column {field.name!r} has nulls")
        t = t.replace_schema_metadata({META_TABLE: table.encode(),
                                       META_VERSION: SCHEMA_VERSION.encode()})

        d = self.table_dir(table) / f"run_id={run_id}"
        d.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        final = d / f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = d / f".{final.name}.tmp"
        pq.write_table(t, tmp, compression="zstd")
        os.replace(tmp, final)  # atomic: readers never see a partial file
        return final

    # ----------------------------------------------------------------- blobs
    def put_blob(self, data: str | bytes) -> str:
        raw = data.encode() if isinstance(data, str) else data
        ref = hashlib.sha256(raw).hexdigest()
        path = self.root / "blobs" / ref[:2] / ref
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(raw)
            os.replace(tmp, path)
        return ref

    def get_blob(self, ref: str) -> bytes:
        return (self.root / "blobs" / ref[:2] / ref).read_bytes()
