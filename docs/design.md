# Design: Agent Research Data Platform

**Status:** Draft v0.1 · **Author:** Alexander Kingson

> Draft written alongside Week 1. Sections marked **TODO** need to be filled in from real use.

## 1. Problem
Research on LLM agents produces several kinds of data that usually end up scattered and inconsistent:
- **transcripts** (multi-turn conversations with tool calls), often in ad hoc JSONL formats that differ by script;
- **training and eval metrics**, split across experiment trackers, logs and notebooks;
- **grades** from automatic metrics, LLM judges and humans, frequently overwritten when re-graded;
- **derived datasets** (filtered trajectories, SFT sets, checkpoints) whose origins are hard to reconstruct.

The result: questions like "which eval set and data produced this checkpoint?", "did any eval question leak into training?" or "how do failure modes differ between architectures?" take hours of archaeology instead of one query.

## 2. Goals and non-goals
**Goals**
1. One canonical, documented data model for agent transcripts that research code writes into directly.
2. A metrics library that training code integrates with one line.
3. Dataset versions with content hashes and lineage, from eval set to checkpoint to results.
4. Data contracts that make the data trustworthy (integrity, contamination).
5. Fast exploratory queries with no infrastructure to run (Parquet + DuckDB on a laptop or one VM).

**Non-goals (for now)**
- Multi-writer concurrency on the catalog; real-time dashboards; access control.
- Replacing a full experiment tracker UI.

## 3. Users and user stories
Primary user: a researcher running agent eval and fine-tuning experiments (initially, me on the companion project).

- As a researcher, I can log an agent episode with its turns, tool calls and grades in one call, without thinking about storage.
- As a researcher, I can compare a metric across runs from the command line.
- As a researcher, I can re-grade episodes (for example, by hand) without losing the original grades.
- As a researcher, I can see the full lineage of a checkpoint.
- As a researcher, registering a training set fails loudly if it overlaps with the eval set.
- **TODO:** add stories discovered during Weeks 2–3 and record which ones changed the design.

## 4. Data model
Tables: `runs`, `metrics`, `episodes`, `turns`, `tool_calls`, `grades` (see `rdp/schemas`).

```
runs 1─* metrics
runs 1─* episodes 1─* turns 1─* tool_calls
                  1─* grades
```

Key decisions:
- **`run_id` on every row.** It is denormalized but nearly free with dictionary encoding. It makes each file self-describing and matches the dominant query, "everything about run X."
- **Large payloads in a blob store.** Turn content above 4 KB and all tool results are stored by sha256 and referenced. Narrow tables keep scans fast; identical tool results are stored once.
- **Append-only status and grades.** `runs` and `grades` keep every version; views `runs` and `grades` select the latest. Corrections never destroy history.
- **Schema version in Parquet file metadata.** Minor versions are additive only (new nullable columns); readers use `union_by_name`. A major bump requires a migration.

## 5. Storage layout
```
<root>/tables/<table>/run_id=<run_id>/part-<ts>-<uuid>.parquet   # zstd, written atomically
<root>/blobs/<sha[:2]>/<sha>
<root>/catalog.duckdb                                            # dataset registry
```
- Files are written to a temp name and renamed, so readers never see partial files.
- Partitioning by run matches the access pattern; read-time pruning is to be evaluated in the Week 4 benchmarks.
- **Known trade-off:** many small files for runs that flush often. Mitigations: larger flush buffers, and a compaction job (TODO, Week 4).

## 6. Metrics library
- `start_run()` writes a `running` status row immediately, so in-flight runs are visible.
- Buffered writes (default 1,000 rows); `finish()`/context-manager exit flushes and marks `ok` or `failed`; an exit hook marks unfinished runs `crashed` and keeps their data.
- `rdp.metrics.hf.rdp_callback(run)` integrates with Hugging Face `Trainer` / TRL trainers.
- Captures `config_json` and the git commit (with a `-dirty` flag).

## 7. Dataset registry and lineage
- `register(name, path, parents, producer_run_id, checks)` creates `<name>:v<n>`, records a content hash, format, row count, schema, commit and producer run, and links parents.
- Idempotent on content: re-registering identical bytes returns the existing version.
- `verify()` detects files modified after registration.
- `lineage()` returns all ancestors through a recursive query.

## 8. Data contracts
Run checks: tool calls reference existing turns; every episode has a grade; non-negative token counts; contiguous turn indexes; `n_turns` matches turns; exactly one content source per turn; schema major version matches.
Dataset checks: `no_overlap` normalizes text and blocks registering a training set that contains eval items.

## 9. Alternatives considered
| Option | Why not (for this scope) |
|---|---|
| MLflow / W&B | Strong for metrics and UI, but no first-class transcript model or contamination checks; adds a server or SaaS dependency. Could be an export target later. |
| Postgres | Good for the catalog, but poor fit for large analytical scans over transcripts; another service to run. |
| Iceberg / Delta / DuckLake | Real table formats with snapshots and compaction. Probably the right answer at larger scale; evaluate DuckLake in Week 4 (ADR to follow). |
| JSONL files | Simple, but no schema enforcement, slow scans and no column pruning. |

## 10. Risks and open questions
- Catalog concurrency: DuckDB allows one writer process. Fine for one researcher; it needs a service or a different catalog backend for a team.
- Small-file growth: needs compaction.
- **TODO:** what researchers actually query most (track it in Weeks 2–3 and promote those queries into `views.sql`).

## 11. Milestones
See the README roadmap.
