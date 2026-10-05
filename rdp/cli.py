"""Command-line interface. The store root comes from --root or $RDP_HOME (default ./rdp_data)."""

from __future__ import annotations

import argparse
import sys

from rdp.contracts import check_run
from rdp.datasets import Registry
from rdp.query import connect
from rdp.store import Store


def _runs(store, args):
    print(connect(store).sql("SELECT run_id, kind, model, status, started_at, updated_at "
                             "FROM runs ORDER BY started_at DESC"))


def _compare(store, args):
    con = connect(store)
    rel = con.sql("SELECT * FROM metric_summary WHERE run_id IN (SELECT unnest($runs)) "
                  "AND ($metric IS NULL OR name = $metric) ORDER BY name, run_id",
                  params={"runs": args.run_ids, "metric": args.metric})
    print(rel)


def _check(store, args):
    results = check_run(store, args.run_id)
    for r in results:
        print(r)
    return 0 if all(r.passed for r in results) else 1


def _datasets(store, args):
    for row in Registry(store).list():
        print(*row, sep="\t")


def _lineage(store, args):
    for version_id, depth, producer in Registry(store).lineage(args.version_id):
        print(f"{'  ' * depth}{version_id}" + (f"  (run {producer})" if producer else ""))


def _show(store, args):
    con = connect(store)
    print(con.sql("SELECT * FROM episodes WHERE episode_id = $e", params={"e": args.episode_id}))
    turns = con.execute("SELECT turn_idx, role, content, content_ref FROM turns "
                        "WHERE episode_id = ? ORDER BY turn_idx", [args.episode_id]).fetchall()
    calls = con.execute("SELECT turn_idx, tool_name, args_json, result_ref FROM tool_calls "
                        "WHERE episode_id = ? ORDER BY turn_idx, call_idx",
                        [args.episode_id]).fetchall()
    for idx, role, content, ref in turns:
        text = content if content is not None else store.get_blob(ref).decode()
        print(f"\n[{idx}] {role}: {text[:args.width]}")
        for t, tool, a, rref in calls:
            if t == idx:
                result = store.get_blob(rref).decode() if rref else ""
                print(f"    -> {tool}({a}) = {result[:args.width]}")
    print(con.sql("SELECT grader, score, label, failure_mode, graded_at FROM grades "
                  "WHERE episode_id = $e", params={"e": args.episode_id}))


def _sql(store, args):
    print(connect(store).sql(args.query))


def _demo(store, args):
    from rdp.demo import build_demo

    build_demo(store, n_episodes=args.episodes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rdp", description=__doc__)
    p.add_argument("--root", help="store root (default: $RDP_HOME or ./rdp_data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("runs", help="list runs with latest status").set_defaults(fn=_runs)
    c = sub.add_parser("compare", help="compare metric summaries across runs")
    c.add_argument("run_ids", nargs="+")
    c.add_argument("--metric")
    c.set_defaults(fn=_compare)
    c = sub.add_parser("check", help="run data-contract checks on a run")
    c.add_argument("run_id")
    c.set_defaults(fn=_check)
    sub.add_parser("datasets", help="list registered dataset versions").set_defaults(fn=_datasets)
    c = sub.add_parser("lineage", help="show ancestors of a dataset version")
    c.add_argument("version_id")
    c.set_defaults(fn=_lineage)
    c = sub.add_parser("show", help="print one episode transcript")
    c.add_argument("episode_id")
    c.add_argument("--width", type=int, default=300)
    c.set_defaults(fn=_show)
    c = sub.add_parser("sql", help="run SQL against the platform views")
    c.add_argument("query")
    c.set_defaults(fn=_sql)
    c = sub.add_parser("demo", help="populate the store with synthetic demo data")
    c.add_argument("--episodes", type=int, default=50)
    c.set_defaults(fn=_demo)

    args = p.parse_args(argv)
    return args.fn(Store(args.root), args) or 0


if __name__ == "__main__":
    sys.exit(main())
