from datetime import UTC, datetime

import pytest

from rdp import connect, start_run
from rdp.metrics import INLINE_LIMIT
from rdp.store import SchemaError, Store


def test_metrics_round_trip_and_status(tmp_path):
    store = Store(tmp_path)
    with start_run("sft", model="m", config={"lr": 1e-4}, store=store, flush_every=3) as run:
        for step in range(5):
            run.log({"loss": 10 - step, "lr": 0.1}, step=step)
    con = connect(store)
    assert con.execute("SELECT count(*) FROM metrics").fetchone()[0] == 10
    assert con.execute("SELECT status FROM runs").fetchall() == [("ok",)]
    assert con.execute("SELECT count(*) FROM runs_history").fetchone()[0] == 2
    last = con.execute("SELECT last_value FROM metric_summary WHERE name = 'loss'").fetchone()
    assert last == (6.0,)


def test_exception_marks_run_failed_but_keeps_data(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(RuntimeError), start_run("eval", store=store) as run:
        run.log({"x": 1}, step=0)
        raise RuntimeError("boom")
    con = connect(store)
    assert con.execute("SELECT status FROM runs").fetchone() == ("failed",)
    assert con.execute("SELECT count(*) FROM metrics").fetchone() == (1,)


def test_log_episode_offloads_long_content_and_tool_results(tmp_path):
    store = Store(tmp_path)
    long_text = "x" * (INLINE_LIMIT + 1)
    with start_run("eval", store=store) as run:
        ep = run.log_episode(
            turns=[{"role": "user", "content": "q"},
                   {"role": "assistant", "content": long_text},
                   {"role": "assistant", "content": "answer"}],
            tool_calls=[{"turn_idx": 1, "tool_name": "search", "args": {"q": "a"},
                         "result": "r1"},
                        {"turn_idx": 1, "tool_name": "open_doc", "result": "r2"}],
            architecture="A", tokens_in=10, tokens_out=5)
        run.log_grade(ep, "exact", 1.0)
    con = connect(store)
    content, ref = con.execute(
        "SELECT content, content_ref FROM turns WHERE turn_idx = 1").fetchone()
    assert content is None and store.get_blob(ref).decode() == long_text
    calls = con.execute("SELECT call_idx, tool_name, result_ref FROM tool_calls "
                        "ORDER BY call_idx").fetchall()
    assert [(c[0], c[1]) for c in calls] == [(0, "search"), (1, "open_doc")]
    assert store.get_blob(calls[1][2]) == b"r2"
    assert con.execute("SELECT n_turns FROM episodes").fetchone() == (3,)


def test_regrade_latest_wins_history_kept(tmp_path):
    store = Store(tmp_path)
    with start_run("eval", store=store) as run:
        ep = run.log_episode(turns=[{"role": "user", "content": "q"}])
        run.log_grade(ep, "human", 0.0)
        run.log_grade(ep, "human", 1.0, label="regraded")
    con = connect(store)
    assert con.execute("SELECT score, label FROM grades").fetchall() == [(1.0, "regraded")]
    assert con.execute("SELECT count(*) FROM grades_history").fetchone() == (2,)


def test_unknown_episode_field_fails_at_call_site(tmp_path):
    run = start_run("eval", store=Store(tmp_path))
    with pytest.raises(SchemaError, match="unknown columns"):
        run.log_episode(turns=[], not_a_column=1)
    run.finish()


def test_crash_handler_marks_run_crashed(tmp_path):
    store = Store(tmp_path)
    run = start_run("eval", store=store)
    run.log({"x": 1}, step=0)
    run._atexit()  # what the interpreter calls on exit without finish()
    con = connect(store)
    assert con.execute("SELECT status FROM runs").fetchone() == ("crashed",)
    assert con.execute("SELECT count(*) FROM metrics").fetchone() == (1,)


def test_empty_store_views_are_queryable(tmp_path):
    con = connect(Store(tmp_path))
    for view in ["runs", "metrics", "episodes", "turns", "tool_calls", "grades",
                 "run_accuracy", "failure_modes", "metric_summary"]:
        assert con.execute(f"SELECT count(*) FROM {view}").fetchone() == (0,)


def test_files_from_older_minor_schema_stay_queryable(tmp_path):
    """Schema 1.0 episodes (no stop_reason) and 1.1 episodes read together."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from rdp.schemas import SCHEMAS

    store = Store(tmp_path)
    old_schema = pa.schema([f for f in SCHEMAS["episodes"] if f.name != "stop_reason"])
    old = pa.Table.from_pylist([{"run_id": "old", "episode_id": "e0", "n_turns": 0,
                                 "started_at": datetime.now(UTC)}], schema=old_schema)
    d = store.table_dir("episodes") / "run_id=old"
    d.mkdir(parents=True)
    pq.write_table(old.replace_schema_metadata({b"rdp.schema_version": b"1.0"}),
                   d / "part-old.parquet")
    with start_run("eval", store=store) as run:
        run.log_episode(turns=[], stop_reason="answered")
    rows = connect(store).execute(
        "SELECT run_id = 'old', stop_reason FROM episodes ORDER BY 1").fetchall()
    assert rows == [(False, "answered"), (True, None)]
