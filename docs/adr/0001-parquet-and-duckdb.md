# ADR 0001: Parquet files + DuckDB as the storage and query layer

**Status:** Accepted · **Date:** 2026-10-04

## Context
The platform must store transcripts, metrics and grades for exploratory research. It must be queryable with SQL, run without infrastructure, and stay fast as data grows to tens or hundreds of millions of rows.

## Decision
Store all tables as append-only, zstd-compressed Parquet files partitioned by run, with explicit Arrow schemas. Query them with DuckDB views created at connect time. Keep the dataset registry in a DuckDB database file.

## Consequences
- Pro: no server; columnar scans and predicate pushdown; any tool that reads Parquet (Spark, BigQuery, pandas, Polars) can consume the data.
- Pro: schema enforcement at write time through Arrow.
- Con: single-writer catalog; no snapshots or time travel; small-file growth needs compaction.
- Revisit when there are several concurrent writers or more than about 1B rows (candidate: DuckLake or Iceberg).
