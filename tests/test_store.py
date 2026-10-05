from datetime import UTC, datetime

import pyarrow.parquet as pq
import pytest

from rdp.schemas import SCHEMA_VERSION
from rdp.store import SchemaError, Store

NOW = datetime.now(UTC)


def metric(run_id="r1", **kw):
    return {"run_id": run_id, "step": 1, "name": "loss", "value": 1.0, "ts": NOW, **kw}


def test_write_creates_partitioned_file_with_version_metadata(tmp_path):
    store = Store(tmp_path)
    path = store.write("metrics", [metric()], "r1")
    assert path.parent.name == "run_id=r1"
    meta = pq.read_schema(path).metadata
    assert meta[b"rdp.schema_version"].decode() == SCHEMA_VERSION
    assert meta[b"rdp.table"] == b"metrics"
    assert store.files("metrics", "r1") == [path]


def test_write_empty_is_noop(tmp_path):
    assert Store(tmp_path).write("metrics", [], "r1") is None


@pytest.mark.parametrize("row, msg", [
    (metric(extra=1), "unknown columns"),
    (metric(value=None), "required column 'value'"),
    (metric(run_id="other"), "run_id"),
    (metric(step="not-an-int"), "metrics"),
])
def test_write_rejects_invalid_rows(tmp_path, row, msg):
    with pytest.raises(SchemaError, match=msg):
        Store(tmp_path).write("metrics", [row], "r1")
    assert Store(tmp_path).files("metrics") == []


def test_blobs_are_content_addressed(tmp_path):
    store = Store(tmp_path)
    ref = store.put_blob("hello")
    assert store.put_blob(b"hello") == ref
    assert store.get_blob(ref) == b"hello"
