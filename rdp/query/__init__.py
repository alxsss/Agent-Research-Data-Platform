"""DuckDB query layer over the Parquet store.

``connect(store)`` returns an in-memory DuckDB connection with one view per stored
table (``runs_history``, ``metrics``, ``episodes``, ``turns``, ``tool_calls``,
``grades_history``) plus the curated views in ``views.sql``.
"""

from __future__ import annotations

from importlib.resources import files

import duckdb

from rdp.schemas import SCHEMAS
from rdp.store import Store

# Tables whose raw view is the full append-only history; views.sql derives "latest".
HISTORY_NAMES = {"runs": "runs_history", "grades": "grades_history"}


def connect(store: Store) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for table, schema in SCHEMAS.items():
        name = HISTORY_NAMES.get(table, table)
        if store.files(table):
            # CREATE VIEW can't take bound parameters; quote the path as a SQL literal.
            glob = store.table_glob(table).replace("'", "''")
            con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{glob}', "
                "union_by_name=true, hive_partitioning=false)"
            )
        else:
            con.register(f"_empty_{table}", schema.empty_table())
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM _empty_{table}")
    con.execute(files("rdp.query").joinpath("views.sql").read_text())
    return con
